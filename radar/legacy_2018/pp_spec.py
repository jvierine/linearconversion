#!/usr/bin/env python
#
# Experimental 12-meter range resolution sporadic E-experiment
#
import numpy as n
import matplotlib.pyplot as plt
import sys
sys.path.append("/home/vierinen/lib64/python/")

import digital_rf_deprecated_hdf5 as drf
import stuffr
import h5py
import time
import scipy.fftpack as fft
import os

from mpi4py import MPI
comm = MPI.COMM_WORLD
print("Hello! I'm rank %d from %d running in total..." % (comm.rank, comm.size))

out_dname="/data2/processing/results_pp"
d=drf.read_hdf5("/data2/2018.05.04")
b=d.get_bounds("gr")

def get_tx_vec(z0,current_ipp,rmax,dec=25):
    z0_tx=n.copy(z0)
    z0_tx=z0_tx[0:(rmax*(25/float(dec)))]
    z0_tx[(46*(25/float(dec))):len(z0_tx)]=0.0
    return(z0_tx)

def get_rx_vec(z0,current_ipp,rmax,dec=25):
    z0_rx=n.copy(z0)
    n_pulses=int(n.ceil(rmax/current_ipp))
    
    for pi in range(n_pulses):
        if pi*current_ipp < rmax:
            z0_rx[n.min([(25/float(dec))*rmax,(25/float(dec))*(pi*current_ipp)]):n.min([(25/float(dec))*rmax,(25/float(dec))*(pi*current_ipp+44+190+1)])]=0.0
    return(z0_rx)
              
dec=25L

rmax=2000L

rg_max=int(25L*rmax/float(dec))


def pp_spec(ipp_idxs,current_ipp):
    csp_t0=n.min(ipp_idxs)
    csp_t1=n.max(ipp_idxs)
    if len(ipp_idxs)<1000:
        print("not enough")
        return

    n_ipp=n.min([len(ipp_idxs),10000])
    fdec=100
    rd_spec=n.zeros([rg_max,n_ipp/fdec],dtype=n.float32)
    rd_amps=n.zeros([rg_max,n_ipp],dtype=n.complex64)

    fname="%s/pp_spec-%06d.h5"%(out_dname,csp_t0/1e6)
    if os.path.exists(fname):
        print("File %s exists. Skipping"%(fname))
        return(0)
    
    cput0=time.time()
    
#    n_blocks=len(ipp_idxs)/1000                    
    for i in range(n_ipp):
        print(i)
        i0=ipp_idxs[i]
        z0=stuffr.decimate(d.read_vector_c81d(i0*25L-25L,rmax*25L,"gr"),dec=dec)
        z0_tx=get_tx_vec(z0,current_ipp,rmax,dec=dec)
        z0_rx=get_rx_vec(z0,current_ipp,rmax,dec=dec)
        echo=n.fft.ifft(n.conj(n.fft.fft(z0_tx))*n.fft.fft(z0_rx))
        rd_amps[:,i]=echo
    print("done")
    for ri in range(rd_amps.shape[0]):
        rd_spec[ri,:]=stuffr.decimate(n.fft.fftshift(n.abs(n.fft.fft(rd_amps[ri,:]))**2.0),dec=fdec)
#        if ri%10 == 0:
 #           plt.plot(10.0*n.log10(rd_spec[ri,:])+ri,color="black")
 #   plt.show()

    rv=n.arange(rg_max)*0.15/(25.0/dec)
    freq=n.fft.fftshift(n.fft.fftfreq(n_ipp/fdec,d=(current_ipp/1e6)))
    #  print(rd_spec.shape)
  #  print(rd_amps.shape)

    r_idx=n.where( (rv>50) & (rv < 150) )[0]

    dB=10.0*n.log10(rd_spec)
    nfloor=n.nanmedian(dB)
    dB=dB-nfloor
    dB=dB[r_idx,:]
    plt.pcolormesh(freq,rv[r_idx],dB,vmin=-3,vmax=10)
    plt.colorbar()
    plt.show()
    
    fname="%s/pp_spec-%06d.h5"%(out_dname,csp_t0/1e6)
    print("saving %s"%(fname))
    
#    ho=h5py.File(fname,"w")
 #   ho["t0"]=csp_t0/1e6
  #  ho["t1"]=csp_t1/1e6
  #  ho["ipp"]=current_ipp
  #  ho["rd_spec"]=csp_acf
  #  ho.close()


h=h5py.File("intervals.h5","r")
ipp=n.copy(h["ipp"].value)
txs=n.copy(h["txs"].value)
nipp=n.copy(h["nipp"].value)
h.close()
print(n.unique(ipp))
clp_idx=n.where(ipp != 10000)[0]
ipp=ipp[clp_idx]
txs=txs[clp_idx]
nipp=nipp[clp_idx]
n_intervals=len(ipp)

for i in n.arange(comm.rank,n_intervals,comm.size):
    print("rank %d ipp %d n_ipps %d i0 %d"%(comm.rank,ipp[i],nipp[i],txs[i]))
    ipps=txs[i]+n.arange(nipp[i]-10,dtype=n.int)*ipp[i]
    pp_spec(ipps,ipp[i])
