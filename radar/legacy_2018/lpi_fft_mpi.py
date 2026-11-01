#!/usr/bin/env python

import numpy as n
import matplotlib.pyplot as plt
import sys
sys.path.append("/home/vierinen/lib64/python/")

import digital_rf_hdf5 as drf
import stuffr
import h5py
import time

from mpi4py import MPI
comm = MPI.COMM_WORLD
print("Hello! I'm rank %d from %d running in total..." % (comm.rank, comm.size))

out_dname="results3"
d=drf.read_hdf5("/media/USRPJUHA/eregion/")
b=d.get_bounds("gr")


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

clp_idx=0
clp_t0=0
clp_t1=0

csp_t0=0
csp_t1=0

def lpi_csp_ipps(ipp_idxs,current_ipp):
    csp_t0=n.min(ipp_idxs)
    csp_t1=n.max(ipp_idxs)
    csp_rx_lags[:,:]=0.0
    csp_tx_lags[:,:]=0.0
    cput0=time.time()

    for i in range(len(ipp_idxs)):
        i0=ipp_idxs[i]
        cput1=time.time()
        print("%d ipp %d dt=%1.2f rx val %1.2f tx val %1.2f n_ipp %d"%(i0,current_ipp,cput1-cput0,n.max(n.abs(csp_rx_lags)),n.max(n.abs(csp_tx_lags)),i))
        cput0=time.time()

        z0=stuffr.decimate(d.read_vector_c81d(i0*25L-25L,rmax*25L,"gr"),dec=25)
            
        z0_tx=get_tx_vec(z0,current_ipp,rmax)
        z0_rx=get_rx_vec(z0,current_ipp,rmax)            
        z0_rx[(rmax/2):rmax]=0.0

        for lag_i in range(n_csp_lags-1):
            csp_mlags[lag_i]=lag_i*current_ipp
            z1=stuffr.decimate(d.read_vector_c81d(i0*25L-25L+25L*current_ipp*lag_i,rmax*25L,"gr"),dec=25)
            
            z1_tx=get_tx_vec(z1,current_ipp,rmax)
            z1_rx=get_rx_vec(z1,current_ipp,rmax)

            z1_rx[(rmax/2):rmax]=0.0
                
            # fractional lags
            f_lags=n.arange(0,40)
            if lag_i == 0:
                f_lags=n.arange(4,40)
            
            for fi in range(len(f_lags)):
                lag=f_lags[fi]
                z1f_tx=n.copy(z1_tx)
                z1f_rx=n.copy(z1_rx)
                z1f_tx=n.roll(z1f_tx,lag)
                z1f_rx=n.roll(z1f_rx,lag)
                
                txz=stuffr.decimate(z0_tx*n.conj(z1f_tx),dec=rdec)/1e7
                rxz=stuffr.decimate(z0_rx*n.conj(z1f_rx),dec=rdec)/1e6
                rxvar=n.max(n.abs(rxz))

                T=n.fft.fft(txz)
                R=n.fft.fft(rxz)
                csp_rx_lags[:,lag_i]+=n.conj(T)*(1.0/rxvar)*R
                csp_tx_lags[:,lag_i]+=(1.0/rxvar)*n.abs(T)**2.0
        if i % 100 == 0:
            for lag_i in range(n_csp_lags-1):
                csp_acf[:,lag_i]=n.fft.ifft((1.0/(csp_tx_lags[:,lag_i]))*csp_rx_lags[:,lag_i])
            plt.clf()
            plt.subplot(211)
            plt.title("n_ipp %d ipp %d"%(i,current_ipp))
            plt.plot(csp_acf[:,0].real)
            plt.plot(csp_acf[:,0].imag)
            plt.subplot(212)
            plt.plot(csp_acf[:,1].real)
            plt.plot(csp_acf[:,1].imag)
            plt.savefig("ppacf-%06d.png"%(current_ipp))
            plt.close()
                
    # deconvolve, and save result.
    print("save short ipp result")
    for lag_i in range(n_csp_lags-1):
        csp_acf[:,lag_i]=n.fft.ifft((1.0/(csp_tx_lags[:,lag_i]))*csp_rx_lags[:,lag_i])
    
    fname="%s/csp_acf-%06d.h5"%(out_dname,csp_t0/1e6)
    print("saving %s"%(fname))

    ho=h5py.File(fname,"w")
    ho["t0"]=csp_t0/1e6
    ho["t1"]=csp_t1/1e6
    ho["lags"]=csp_mlags
    ho["ranges"]=n.arange(rmax/rdec)*0.15*rdec
    ho["acf"]=csp_acf
    ho.close()
    
    plt.clf()
    plt.subplot(311)
    plt.plot(csp_acf[:,0].real)
    plt.plot(csp_acf[:,0].imag)
    plt.title("lag %d dt %1.2f"%(current_ipp*0,(csp_t1-csp_t0)/1e6))
    
    plt.subplot(312)
    plt.plot(csp_acf[:,1].real)
    plt.plot(csp_acf[:,1].imag)
    plt.title("lag %d"%(current_ipp))
    
    plt.subplot(313)
    plt.plot(csp_acf[:,2].real)
    plt.plot(csp_acf[:,2].imag)
    plt.title("lag %d"%(current_ipp*2))
    
    plt.savefig("%s/csp_acf-%06d.png"%(out_dname,csp_t0/1e6))
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
    lpi_csp_ipps(ipps,ipp[i])
