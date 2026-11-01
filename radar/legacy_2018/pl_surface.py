#!/usr/bin/env python

import glob
import numpy as n
import matplotlib.pyplot as plt
import h5py
import stuffr


dname="results4"

fl=glob.glob("%s/*pl*.h5"%(dname))
fl.sort()

nf=len(fl)
P=n.zeros([nf,400])
F=n.zeros([nf,400])
alpha=42859649122.80702
f0=-7
f1=-3
tv=n.zeros(nf)
for fi,f in enumerate(fl):
    print(f)
    h=h5py.File(f,"r")
#    print(h.keys())
    PL=n.copy(h["PL"].value)
    tv[fi]=h["t0"].value
    freqs=h["freqs"].value

    Z=n.copy(PL)
    Z[:,:]=0.0
 #   print(Z.shape)
    Z[(-10+Z.shape[0]/2):(10+Z.shape[0]/2),(-10+Z.shape[1]/2):(10+Z.shape[1]/2)]=1.0/(21*21)
    SM=n.fft.fftshift(n.fft.ifft2(n.fft.fft2(PL)*n.fft.fft2(Z)))
    hpf=PL-SM.real

    nfloor=n.median(n.abs(hpf))
    hpf=(hpf-nfloor)/nfloor

    
    PL=hpf

    if PL.shape[1] > 1600:
        PL=PL/3.33  # scale to 44 us SNR
        
    print(freqs)
    print(P.shape)
    fidx=n.where((freqs>f0)&(freqs<f1))[0]
#    plt.pcolormesh(PL[:,fidx],vmin=0,vmax=10)
#    plt.colorbar()
#    plt.show()
    for ri in range(400):
        mfi=n.argmax(PL[ri,fidx])
        if mfi == len(fidx)-1:
            mfi = mfi-1
        if mfi == 0:
            mfi = 1
        
        P[fi,ri]=(1.0/3.0)*PL[ri,fidx[mfi-1]]+PL[ri,fidx[mfi]]+PL[ri,fidx[mfi+1]]
        w0=PL[ri,fidx[mfi-1]]
        w1=PL[ri,fidx[mfi]]
        w2=PL[ri,fidx[mfi+1]]        
        F[fi,ri]=(1.0/(w0+w1+w2))*(w0*freqs[fidx[mfi-1]]+w1*freqs[fidx[mfi]]+w2*freqs[fidx[mfi+1]])
        
    ofname="results4/ext-%06d.h5"%(h["t0"].value)
    print("writing %s"%(ofname))
    ho=h5py.File(ofname,"w")
    ho["t0"]=h["t0"].value
    ho["P"]=P[fi,:]
    ho["F"]=F[fi,:]
    ho["mode"]=PL.shape[1]
    ho.close()
