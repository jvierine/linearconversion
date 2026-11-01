#!/usr/bin/env python

import numpy as n
import matplotlib.pyplot as plt
import sys
sys.path.append("/home/vierinen/lib64/python/")

import digital_rf_hdf5 as drf
import stuffr
import h5py
import time
import pyfftw
import scipy.fftpack as fft

pyfft=pyfftw.interfaces.numpy_fft.fft
pyifft=pyfftw.interfaces.numpy_fft.ifft


from mpi4py import MPI
comm = MPI.COMM_WORLD
print("Hello! I'm rank %d from %d running in total..." % (comm.rank, comm.size))

def get_tx_vec(z0,current_ipp,rmax):
    z0_tx=n.copy(z0)
    z0_tx=z0_tx[0:rmax]
    z0_tx[46:(rmax)]=0.0
    return(z0_tx)

def get_rx_vec(z0,current_ipp,rmax):
    z0_rx=n.copy(z0)
    n_pulses=int(n.ceil(rmax/current_ipp))

    for pi in range(n_pulses):
        if pi*current_ipp < rmax:
            z0_rx[n.min([rmax,pi*current_ipp]):n.min([rmax,pi*current_ipp+44+190+1])]=0.0
    return(z0_rx)
              
d=drf.read_hdf5("/media/USRPJUHA/eregion/")
b=d.get_bounds("gr")




rdec=2L
rmax=10000L

rg_max=int(rmax/rdec)

frac_lags=n.arange(40)

clp_lags=[]
clp_mlags=[]
for i in range(43):
    if i==0:
        clp_lags.append(n.arange(4,10,dtype=n.int))
        clp_mlags.append(n.mean(n.arange(4,10)))
    else:
        clp_lags.append(n.arange(10,dtype=n.int)+i*10)
        clp_mlags.append(n.mean(n.arange(10)+i*10))
        
n_clp_lags=len(clp_lags)

clp_acf=n.zeros([rg_max,n_clp_lags],dtype=n.complex128)

clp_tx_lags=n.zeros([rg_max,n_clp_lags],dtype=n.complex128)
clp_rx_lags=n.zeros([rg_max,n_clp_lags],dtype=n.complex128)

n_csp_lags=6
csp_acf=n.zeros([rg_max,n_csp_lags],dtype=n.complex128)
csp_mlags=n.zeros(n_csp_lags)
csp_tx_lags=n.zeros([rg_max,n_csp_lags],dtype=n.complex128)
csp_rx_lags=n.zeros([rg_max,n_csp_lags],dtype=n.complex128)

n_clp_int=1000

clp_t0=0
clp_t1=0

csp_t0=0
csp_t1=0

def lpi_csp_ipps(ipp_idxs,current_ipp):
    clp_t0=n.min(ipp_idxs)
    clp_t1=n.max(ipp_idxs)
    clp_rx_lags[:,:]=0.0
    clp_tx_lags[:,:]=0.0

#    txz=pyfftw.n_byte_align_empty(rmax/rdec, dtype='complex64', n=16)
 #   rxz=pyfftw.n_byte_align_empty(rmax/rdec, dtype='complex64', n=16)

    cput0=time.time()
    clp_idx=0
    for i in range(len(ipp_idxs)):
        i0=ipp_idxs[i]
        cput1=time.time()
        print("%d ipp %d dt=%1.2f rx val %1.2f tx val %1.2f n_ipp %d"%(i0,current_ipp,cput1-cput0,n.max(n.abs(clp_rx_lags)),n.max(n.abs(clp_tx_lags)),i))
        cput0=time.time()
        
        # clp, do normal lags
        z0=stuffr.decimate(d.read_vector_c81d(i0*25L-25L,rmax*25L,"gr"),dec=25)
 #       plt.plot(z0.real)
#        plt.show()
        z0_tx=n.copy(z0)
        z0_rx=n.copy(z0)
        z0_tx[460:len(z0_tx)]=0.0  # rx blank
        z0_rx[0:(440+190+2)]=0.0   # tx blank
        clp_idx+=1

        for lag_i in range(len(clp_lags)):
            f_lags=clp_lags[lag_i]
            for flag_i in range(len(f_lags)):
                lag=f_lags[flag_i]
                z1_tx=n.copy(z0_tx)
                z1_rx=n.copy(z0_rx)
                z1_tx=n.roll(z1_tx,lag)
                z1_rx=n.roll(z1_rx,lag)
                
                txz=stuffr.decimate(z0_tx*n.conj(z1_tx),dec=rdec)/1e7
                rxz=stuffr.decimate(z0_rx*n.conj(z1_rx),dec=rdec)/1e6
                rxvar=n.mean(n.abs(rxz))
                T=fft.fft(txz)
                R=fft.fft(rxz)
                clp_rx_lags[:,lag_i]+=n.conj(T)*(1.0/rxvar)*R
                clp_tx_lags[:,lag_i]+=(1.0/rxvar)*n.abs(T)**2.0
                
    
    # Maximum likelihood estimate of each lag with FFT
    for lag_i in range(len(clp_lags)):
        clp_acf[:,lag_i]=n.fft.ifft((1.0/(clp_tx_lags[:,lag_i]))*clp_rx_lags[:,lag_i])

    fname="clp_acf-%06d.h5"%(clp_t0/1e6)
    print("saving %s"%(fname))
    ho=h5py.File("results3/clp_acf-%06d.h5"%(clp_t0/1e6),"w")
    ho["t0"]=clp_t0/1e6
    ho["t1"]=clp_t1/1e6
    ho["lags"]=clp_mlags
    ho["n_ipp"]=clp_idx
    ho["ranges"]=n.arange(rmax/rdec)*0.15*rdec
    ho["acf"]=clp_acf
    ho.close()
    # zero stuff
    clp_idx=0
    
    plt.clf()
    plt.pcolormesh(clp_acf.real)
    plt.colorbar()
    plt.savefig("results3/clp_acf-%06d.png"%(clp_t0/1e6))
    plt.close()

h=h5py.File("intervals.h5","r")
ipp=n.copy(h["ipp"].value)
txs=n.copy(h["txs"].value)
nipp=n.copy(h["nipp"].value)
h.close()

clp_idx=n.where(ipp == 10000)[0]
ipp=ipp[clp_idx]
txs=txs[clp_idx]
nipp=nipp[clp_idx]
n_intervals=len(ipp)

for i in n.arange(comm.rank,n_intervals,comm.size):
    
    print("rank %d ipp %d n_ipps %d i0 %d"%(comm.rank,ipp[i],nipp[i],txs[i]))
    ipps=txs[i]+n.arange(nipp[i]-2,dtype=n.int)*ipp[i]
    lpi_csp_ipps(ipps,ipp[i])

    
