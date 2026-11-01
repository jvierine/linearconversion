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
    n_range_gates=100
    range_gates=n.array(25L*n.linspace(533,933,num=n_range_gates),dtype=n.int64)

    txlen=(44L+1000L)*25L 
    nfft=2**15
    freqs=n.fft.fftshift(n.fft.fftfreq(nfft,d=1.0/25e6))/1e6
    ranges=(range_gates/25L)*0.15
    tx_idx=n.arange(txlen)
    sr=25L
    SCLP = n.zeros([n_range_gates, nfft],dtype=n.float64)
    offset=0
    SCLP[:,:]=0.0
    rgmax=933
    
    for i in range(len(ipps)):
        print(i)
        i0=ipps[i]*25L
        z=d.read_vector_c81d(i0+offset,2*ipp*25L,channel)
        ztx=n.copy(n.conj(z[0:txlen]))
        ztx[(25L*46L):(25L*1000L)]=0.0
#        ztx[(25L*(1000L+46L)):len(ztx)]=0.0
 #       plt.plot(ztx.real)
  #      plt.plot(ztx.imag)
   #     plt.show()
        for ri,rg in enumerate(range_gates):
            SCLP[ri,:]+=n.fft.fftshift(n.abs(scipy.fftpack.fft(ztx*z[rg+tx_idx],n=nfft))**2.0)

#    for i in range(SCLP.shape[1]):
 #       noise=n.mean(SCLP[:,i])
  #      SCLP[:,i]=(SCLP[:,i]-noise)/noise
    plt.pcolormesh(freqs,ranges,SCLP)
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Range (km)")
    plt.colorbar()
    plt.show()

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
