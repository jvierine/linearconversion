#!/usr/bin/env python

import glob
import numpy as n
import matplotlib.pyplot as plt
import h5py
import stuffr


dname="results5"

fl=glob.glob("%s/spl*.h5"%(dname))
fl.sort()

nf=len(fl)

for f in fl:
    print(f)

    h=h5py.File(f,"r")
    SCLP=n.copy(h["PL"].value)
    SCLP2=n.copy(SCLP)
    for i in range(SCLP.shape[1]):
        SCLP2[:,i]=SCLP[:,i]-n.mean(SCLP[0:10,i])

    ranges=h["range"].value
    freqs=h["freqs"].value
    t0=h["t0"].value/1e6
    i0=h["t0"].value

    if os.path.exists("%s/lpspl-%d.png"%(dname,i0)):
        print("Plot for %s already exists"%(f))
        continue
    #t0=h["t0"].value/1e6

#    hacf=h5py.File("%s/scp_acf-%06d.h5"%(dname),"r")
 #   hacf.close()


 #  plt.clf()


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
    plt.pcolormesh(freqs,ranges,hpf.real,vmin=0,vmax=1e14)
    plt.title("%s"%(stuffr.unix2datestr(t0)))
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Range (km)")
    plt.xlim([-1,11])
    plt.ylim([80,140])
    plt.colorbar()
    plt.subplot(212)
    plt.pcolormesh(freqs,ranges,hpf.real,vmin=0,vmax=1e14)
    plt.title("%s"%(stuffr.unix2datestr(t0)))
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Range (km)")
    plt.xlim([-11,1])
    plt.ylim([80,140])
    plt.colorbar()

    plt.tight_layout()
#    plt.show()
    plt.savefig("%s/lpspl-%d.png"%(dname,i0))
    plt.close()
    h.close()
