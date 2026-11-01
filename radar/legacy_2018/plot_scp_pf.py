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

P=n.zeros([nf,5000],dtype=n.complex)
N=n.zeros([nf,5000],dtype=n.complex)
# magic constant
alpha=43859649122.80702
for fi,f in enumerate(fl):
    print(f)
    h=h5py.File(f,"r")
    print(h.keys())
    S[fi,:]=h["acf"].value[:,0]
    r=n.copy(h["ranges"].value)
#    N[fi,:]=1e12*(S[fi,:]*r**2.0*0.9)

    N[fi,:]=alpha*(S[fi,:]*r**2.0)
    P[fi,:]=8.98*n.sqrt(alpha*(S[fi,:]*r**2.0) )
#    plt.plot(alpha*(S[fi,:]*r**2.0),r)
 #   plt.ylim([0,150])
  #  plt.show()

    tv[fi]=h["t0"].value

    h.close()

#plt.plot(tv,".")
#plt.show()
idx=n.argsort(tv)[::-1]

S=S[idx,:]
N=N[idx,:]
P=P[idx,:]
tv=tv[idx]
rv=n.arange(5000)*0.3
dB=10.0*n.log10(n.abs(S.real))
import stuffr
dts=[]
for t in tv:
    print(t)
    dts.append(stuffr.unix2date(t))
print(len(dts))
print(len(tv))

plt.pcolormesh(dts,rv,n.log10(n.transpose(N)),vmin=7,vmax=13,cmap="jet")
plt.ylim([30,700])
plt.colorbar()
plt.show()

plt.pcolormesh(dts,rv,n.transpose(P),vmin=3e6,vmax=13e6,cmap="jet")
plt.ylim([30,700])
plt.colorbar()
plt.show()


plt.pcolormesh(dts,rv,n.transpose(dB),vmin=-60.0,vmax=-28,cmap="jet")

plt.ylim([30,700])
plt.colorbar()
plt.show()
