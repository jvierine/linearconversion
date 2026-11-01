#!/usr/bin/env

import numpy as n
import matplotlib.pyplot as plt
import h5py
import glob
import stuffr

fl=glob.glob("./results_150m/cs*.h5")

fl.sort()
h=h5py.File(fl[0],"r")
nr=len(h["acf"].value[:,0])
h.close()

nf=len(fl)

#nr=5000
S=n.zeros([nf,nr],dtype=n.complex)

tv=n.zeros(nf)

rv=n.zeros(nr)
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
idx=n.argsort(tv)
S=S[idx,:]
tv=tv[idx]


rv=n.arange(nr)*0.15
dB=10.0*n.log10(n.abs(S.real))

#alpha=42859649122.80702
#for fi in range(S.shape[0]):
#    plt.plot(n.log10(5*alpha*n.real(S[fi,:])*rv**2.0),rv,color="gray")
#    fi2=n.argmin(n.abs(tv300-tv[fi]))
#    plt.plot(n.log10(alpha*n.real(S300[fi2,:])*rv300**2.0),rv300,color="black")
#    plt.title("%f %s"%(tv300[fi2]-tv[fi],stuffr.unix2datestr(tv[fi])))
#    plt.ylim([80,120])
#    plt.xlim([10,12])
#    plt.show()

plt.pcolormesh(tv,rv,n.transpose(dB),vmin=-60.0-5+3,vmax=-30-5+3,cmap="jet")
plt.ylim([30,140])
plt.colorbar()
plt.show()
