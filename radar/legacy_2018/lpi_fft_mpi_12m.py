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

out_dname="./results_150m"
d=drf.read_hdf5("../2018.05.04")
b=d.get_bounds("gr")

def get_tx_vec(z0,current_ipp,rmax,dec=25):
    z0_tx=n.copy(z0)
    z0_tx=z0_tx[0:int((rmax*(25/float(dec))))]
    z0_tx[int((46*(25/float(dec)))):len(z0_tx)]=0.0
    return(z0_tx)

def get_rx_vec(z0,current_ipp,rmax,dec=25):
    z0_rx=n.copy(z0)
    n_pulses=int(n.ceil(rmax/current_ipp))
    
    for pi in range(n_pulses):
        if pi*current_ipp < rmax:
            z0_rx[n.min([int((25/float(dec))*rmax),int((25/float(dec))*(pi*current_ipp))]):n.min([int((25/float(dec))*rmax),int((25/float(dec))*(pi*current_ipp+44+190+1))])]=0.0
    return(z0_rx)
              
rdec=1L
dec=25L

rmax=2000L

rg_max=int(25L*rmax/float(rdec)/float(dec))

n_csp_lags=1
csp_acf=n.zeros([rg_max,n_csp_lags],dtype=n.complex128)
csp_mlags=n.zeros(n_csp_lags)
csp_tx_lags=n.zeros([rg_max,n_csp_lags],dtype=n.complex128)
csp_rx_lags=n.zeros([rg_max,n_csp_lags],dtype=n.complex128)

def lpi_csp_ipps(ipp_idxs,current_ipp):
    csp_t0=n.min(ipp_idxs)
    csp_t1=n.max(ipp_idxs)
    csp_rx_lags[:,:]=0.0
    csp_tx_lags[:,:]=0.0

    
    fname="%s/csp_acf-%06d.h5"%(out_dname,csp_t0/1e6)
    if os.path.exists(fname):
        print("File %s exists. Skipping"%(fname))
        return(0)
    cput0=time.time()
    dratio=(25/float(dec))
                        
    for i in range(len(ipp_idxs)):
        i0=ipp_idxs[i]
 #       cput1=time.time()
#        print("%d ipp %d dt=%1.2f rx val %1.2f tx val %1.2f n_ipp %d"%(i0,current_ipp,cput1-cput0,n.max(n.abs(csp_rx_lags)),n.max(n.abs(csp_tx_lags)),i))
  #      cput0=time.time()

        if dec>1:
            z0=stuffr.decimate(d.read_vector_c81d(i0*25L-25L,rmax*25L,"gr"),dec=dec)
        else:
            z0=d.read_vector_c81d(i0*25L-25L,rmax*25L,"gr")
            
        z0_tx=get_tx_vec(z0,current_ipp,rmax,dec=dec)
        z0_rx=get_rx_vec(z0,current_ipp,rmax,dec=dec)            
        z0_rx[int((dratio*rmax*0.8)):len(z0_rx)]=0.0

        for lag_i in range(n_csp_lags):
            ct0=time.time()
            csp_mlags[lag_i]=lag_i*current_ipp

            if dec>1:
                z1=stuffr.decimate(d.read_vector_c81d(i0*25L-25L+25L*current_ipp*lag_i,rmax*25L,"gr"),dec=dec)
            else:
                z1=d.read_vector_c81d(i0*25L-25L+25L*current_ipp*lag_i,rmax*25L,"gr")
            ct1=time.time()
#            print("read %1.2f"%(ct1-ct0))

            ct0=time.time()
            z1_tx=get_tx_vec(z1,current_ipp,rmax,dec=dec)
            z1_rx=get_rx_vec(z1,current_ipp,rmax,dec=dec)

            z1_rx[int((dratio*rmax*0.8)):len(z1_rx)]=0.0
            ct1=time.time()
 #           print("cleanup %1.2f"%(ct1-ct0))

            # fractional lags
            f_lags=n.arange(0,int(40*(dratio)))
            if lag_i == 0:
                f_lags=n.arange(int(4*(dratio)),int(44*dratio))
#            f_lags=n.arange(4*dratio,44*dratio)
            ct0=time.time()

            # TODO: I have to implement this with GPU

            for fi in range(len(f_lags)):
  #              print(fi)
                lag=f_lags[fi]
                z1f_tx=n.copy(z1_tx)
                z1f_rx=n.copy(z1_rx)
                z1f_tx=n.roll(z1f_tx,lag)
                z1f_rx=n.roll(z1f_rx,lag)
                
                txz=stuffr.decimate(z0_tx*n.conj(z1f_tx),dec=rdec)/1e7
                rxz=stuffr.decimate(z0_rx*n.conj(z1f_rx),dec=rdec)/1e6
                rxvar=n.max(n.abs(rxz))

                T=fft.fft(txz)
                R=fft.fft(rxz)
                csp_rx_lags[:,lag_i]+=n.conj(T)*(1.0/rxvar)*R
                csp_tx_lags[:,lag_i]+=(1.0/rxvar)*n.abs(T)**2.0

            ct1=time.time()
                
    # deconvolve, and save result.
    print("save short ipp result")
    for lag_i in range(n_csp_lags):
#        pwr_spec=csp_tx_lags[:,lag_i]
 #       pwr_spec_dB=10.0*n.log10(n.fft.fftshift(n.real(pwr_spec)))
        csp_acf[:,lag_i]=n.fft.ifft((1.0/(csp_tx_lags[:,lag_i]))*csp_rx_lags[:,lag_i])
    
    fname="%s/csp_acf-%06d.h5"%(out_dname,csp_t0/1e6)
    print("saving %s"%(fname))
    
    ho=h5py.File(fname,"w")
    ho["t0"]=csp_t0/1e6
    ho["t1"]=csp_t1/1e6
    ho["lags"]=csp_mlags
    ho["ranges"]=n.arange(dratio*rmax/rdec)*(0.15/dratio)*rdec
    ho["acf"]=csp_acf
    ho.close()

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
    lpi_csp_ipps(ipps,ipp[i])
