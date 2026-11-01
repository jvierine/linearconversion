#!/usr/bin/env python

import glob
import numpy as n
import matplotlib.pyplot as plt
import h5py
import stuffr

dname="results4"
fl=glob.glob("%s/spl*.h5"%(dname))
fl.sort()

nf=len(fl)
h=h5py.File(fl[0],"r")
SCLP=n.copy(h["PL"].value)
rv=n.copy(h["range"].value)
h.close()
nr=SCLP.shape[0]
nfreq=SCLP.shape[1]
S=n.zeros([nf,nr])
tv=n.zeros(nf)

mpwrs=[]
mfreqs=[]

for fi,f in enumerate(fl):
    print(f)
    print(fi)
    h=h5py.File(f,"r")
    print(h.keys())
    SCLP=n.copy(h["PL"].value)
    SCLP2=n.copy(SCLP)
    for i in range(SCLP.shape[1]):
        SCLP2[:,i]=SCLP[:,i]-n.mean(SCLP[0:10,i])
    print(SCLP2.shape)        
    for i in range(SCLP.shape[0]):
        S[fi,i]=n.max(SCLP2[i,0:(1024-200)])
    ranges=h["range"].value
    freqs=h["freqs"].value
    SCLP[:,(1024-100):(1024+100)]=0.0

    midx=n.unravel_index(n.argmax(SCLP),SCLP.shape)
    pwr=10.0*n.log10(n.max(SCLP))
    mfreqs.append(freqs[midx[1]])
    mpwrs.append(pwr)
    print("%1.2f %1.2f"%(pwr,freqs[midx[1]]))
    t0=h["t0"].value
    tv[fi]=t0

plt.plot(mfreqs,mpwrs,"x")
plt.show()
plt.pcolormesh(tv,rv,10.0*n.log10(n.transpose(S)))
plt.show()

