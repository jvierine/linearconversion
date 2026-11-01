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
#    S[fi,:]=h["acf"].value[:,0]
    acf=n.copy(h["acf"].value)

    tv[fi]=h["t0"].value
    S=n.zeros([acf.shape[0],acf.shape[1]*2])
    S2=n.zeros([acf.shape[0],acf.shape[1]*2])
    for i in range(acf.shape[0]):
        acfn=n.copy(acf[i,:]/n.real(acf[i,0]))
        S2[i,:]=n.fft.fftshift(n.abs(n.fft.fft(acf[i,:],2*acf.shape[1])))
        S[i,:]=S2[i,:]/n.sum(S2[i,:])
    fvec=n.fft.fftshift(n.fft.fftfreq(2*acf.shape[1],d=10e-6))
    rvec=n.copy(h["ranges"].value)
    plt.subplot(121)
    plt.pcolormesh(fvec,rvec,S)
    plt.colorbar()
    plt.title("%s"%(stuffr.unix2datestr(tv[fi])))
    plt.ylim([50,150])
    plt.xlim([-10e3,10e3])
    plt.subplot(122)
    plt.pcolormesh(fvec,rvec,10.0*n.log10(n.abs(S2)))
    plt.colorbar()
    plt.title("%s"%(stuffr.unix2datestr(tv[fi])))
    plt.ylim([50,150])
    plt.xlim([-10e3,10e3])
    plt.savefig("results3/clp_spec-%06d.png"%(tv[fi]))
    plt.close()

    
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
