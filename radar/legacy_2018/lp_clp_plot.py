#!/usr/bin/env python

import glob
import numpy as n
import matplotlib.pyplot as plt
import h5py
import stuffr

dname="results4"
fl=glob.glob("%s/cpl*.h5"%(dname))
fl.sort()

nf=len(fl)
h=h5py.File(fl[0],"r")
SCLP=n.copy(h["PL"].value)
h.close()
nr=SCLP.shape[0]
nfreq=SCLP.shape[1]
S=n.zeros([nf,nr])

for fi,f in enumerate(fl):
    print(f)
    h=h5py.File(f,"r")
    SCLP=n.copy(h["PL"].value)
    SCLP2=n.copy(SCLP)
    for i in range(SCLP.shape[1]):
        SCLP2[:,i]=SCLP[:,i]-n.mean(SCLP[0:10,i])
    ranges=h["range"].value
    freqs=h["freqs"].value
    t0=h["t0"].value

    SCLP2[:,(8192-1000):(8192+1000)]=0.0
    Z=n.copy(SCLP2)
    Z[:,:]=0.0
    #   print(Z.shape)
    Z[(-10+Z.shape[0]/2):(10+Z.shape[0]/2),(-10+Z.shape[1]/2):(10+Z.shape[1]/2)]=1.0/(21*21)
    #    Z=n.fft.fftshift(Z)
    #    Z[(0):(10),(0):(10)]=1.0/(11*11)
    #    plt.pcolormesh(freqs,ranges,Z.real)
    #   plt.show()
    SM=n.fft.fftshift(n.fft.ifft2(n.fft.fft2(SCLP2)*n.fft.fft2(Z)))
    hpf=SCLP2-SM.real
    hpf=hpf[:,::-1]
    nfreq=SCLP2.shape[1]
    #    hpf[:,(nfreq/2):nfreq]=hpf[:,(nfreq/2):nfreq] + hpf[:,(nfreq/2):0:-1]
    
    plt.figure(figsize=(15,2*6.4))
    plt.subplot(211)
    plt.pcolormesh(freqs,ranges,hpf.real,vmin=0,vmax=1.6e15)
    plt.title("%s"%(stuffr.unix2datestr(t0)))
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Range (km)")
    plt.xlim([3,6])
    plt.ylim([80,140])
    plt.colorbar()
    plt.subplot(212)
    plt.pcolormesh(freqs,ranges,hpf.real,vmin=0,vmax=1.6e15)
    plt.title("%s"%(stuffr.unix2datestr(t0)))
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Range (km)")
    plt.xlim([-6,-3])
    plt.ylim([80,140])
    plt.colorbar()
    
    plt.tight_layout()
#    plt.show()
    plt.savefig("%s/cpl-%06d.png"%(dname,t0))
    plt.close()
    h.close()
