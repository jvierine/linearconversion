#!/usr/bin/env python
#
# plasma-line with short pulse
#
import numpy as n
import matplotlib.pyplot as plt
import sys
sys.path.append("/home/vierinen/lib64/python/")

import digital_rf_hdf5 as drf
import stuffr
import scipy
import os
import scipy.signal
import scipy.fftpack
import h5py
import time

from mpi4py import MPI
comm = MPI.COMM_WORLD
print("Hello! I'm rank %d from %d running in total..." % (comm.rank, comm.size))

channel="gr"
out_dname="results3"
d=drf.read_hdf5("/media/USRPJUHA/eregion/")
b=d.get_bounds(channel)


def get_tx_vec(z0,current_ipp,rmax):
    z0_tx=n.copy(z0)
    z0_tx=z0_tx[0:rmax]
    z0_tx[46:(rmax)]=0.0
    return(z0_tx)

def get_rx_vec(z0,current_ipp,rmax):
    z0_rx=n.copy(z0)
    n_pulses=int(n.ceil(rmax/current_ipp))
#    print(n_pulses)
    for pi in range(n_pulses):
        if pi*current_ipp < rmax:
            z0_rx[n.min([rmax,pi*current_ipp]):n.min([rmax,pi*current_ipp+44+190+1])]=0.0
    return(z0_rx)

def pl_csp_ipps(ipps,ipp):

    t0=n.min(ipps)/1e6
    print(stuffr.unix2datestr(t0))
    fname="results3/spl-%06d.h5"%(t0)
    if os.path.exists(fname):
        print("Already analyzed %s"%(fname))
        return

    n_range_gates=100
    range_gates=n.array(25L*n.linspace(533,933,num=n_range_gates),dtype=n.int64)

    txlen=44L*25L
    w=n.array(scipy.signal.hamming(txlen),dtype=n.complex64)
    nfft=2048
    freqs=n.fft.fftshift(n.fft.fftfreq(nfft,d=1.0/25e6))/1e6
    ranges=(range_gates/25L)*0.15
    tx_idx=n.arange(txlen)
    sr=25L
    SCLP = n.zeros([n_range_gates, nfft],dtype=n.float64)
    offset=0
    SCLP[:,:]=0.0

    
    for i in range(len(ipps)):

#        print(i)
        i0=ipps[i]*25L
        try:
            z=d.read_vector_c81d(i0+offset,ipp*25L,channel)
        except:
            print("read error. skipping %d"%(i0))
            return
        ztx=n.conj(z[0:txlen])
        for ri,rg in enumerate(range_gates):
            SCLP[ri,:]+=n.fft.fftshift(n.abs(scipy.fftpack.fft(ztx*z[rg+tx_idx],n=nfft))**2.0)
    SCLP=SCLP/float(len(ipps))
    SCLP2=n.copy(SCLP)
    for i in range(SCLP.shape[1]):
        SCLP2[:,i]=SCLP[:,i]-n.mean(SCLP[0:10,i])

    ho=h5py.File("results3/spl-%06d.h5"%(t0),"w")
    ho["PL"]=SCLP
    ho["t0"]=t0
    ho["range"]=ranges
    ho["freqs"]=freqs
    ho.close()
    plt.clf()
    plt.figure(figsize=(10,6.4))
    plt.pcolormesh(freqs,ranges,SCLP2,vmin=0.0,vmax=1.2e14)
    plt.title("%s"%(stuffr.unix2datestr(t0)))
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Range (km)")
    plt.xlim([-1,12.5])
    plt.ylim([80,140])
    plt.colorbar()
   # plt.show()
    plt.savefig("results3/spl-%06d.png"%(t0))
    plt.close()
    

h=h5py.File("intervals.h5","r")
ipp=n.copy(h["ipp"].value)
txs=n.copy(h["txs"].value)
nipp=n.copy(h["nipp"].value)
h.close()

clp_idx=n.where(ipp != 10000)[0]
ipp=ipp[clp_idx]
txs=txs[clp_idx]
nipp=nipp[clp_idx]
n_intervals=len(ipp)

for i in n.arange(comm.rank,n_intervals,comm.size):
    print("rank %d ipp %d n_ipps %d i0 %d"%(comm.rank,ipp[i],nipp[i],txs[i]))
    ipps=txs[i]+n.arange(nipp[i]-10,dtype=n.int)*ipp[i]
    pl_csp_ipps(ipps,ipp[i])
