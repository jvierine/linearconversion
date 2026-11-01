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
              
d=drf.read_hdf5("/media/USRPJUHA/eregion/")
b=d.get_bounds("gr")

h=h5py.File("tx_ereg.h5","r")

txs=n.copy(h["tx_start"].value)
txe=n.copy(h["tx_end"].value)

print(len(txs))
print(len(txe))
n_ipp=len(txs)

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

clp_acf=n.zeros([rg_max,n_clp_lags],dtype=n.complex)

clp_tx_lags=n.zeros([rg_max,n_clp_lags],dtype=n.complex)
clp_rx_lags=n.zeros([rg_max,n_clp_lags],dtype=n.complex)

n_csp_lags=6
csp_acf=n.zeros([rg_max,n_csp_lags],dtype=n.complex)
csp_mlags=n.zeros(n_csp_lags)
csp_tx_lags=n.zeros([rg_max,n_csp_lags],dtype=n.complex)
csp_rx_lags=n.zeros([rg_max,n_csp_lags],dtype=n.complex)

n_clp_int=1000

clp_idx=0
clp_t0=0
clp_t1=0

csp_t0=0
csp_t1=0

current_ipp=txs[1]-txs[0]
cput0=time.time()
for i in range(n_ipp-5):
    # 25 MHz data
    ipp=txs[i+1]-txs[i]
    ipp=10.0*int(ipp/10)
    i0=txs[i]
    cput1=time.time()
    print("%d ipp %d dt=%1.2f"%(i0,ipp,cput1-cput0))
    cput0=time.time()
    # if ipp changes, save result
    if ipp != current_ipp:
        print("ipp from %d changed to %d, writing results"%(current_ipp,ipp))
        if current_ipp < 9000:
            csp_t1=i0
            print("save short ipp result")
            for lag_i in range(n_csp_lags-1):
                csp_acf[:,lag_i]=n.fft.ifft((1.0/(csp_tx_lags[:,lag_i]))*csp_rx_lags[:,lag_i])

            fname="csp_acf-%06d.h5"%(csp_t0/1e6)
            print("saving %s"%(fname))
            ho=h5py.File("results/csp_acf-%06d.h5"%(csp_t0/1e6),"w")
            ho["t0"]=csp_t0/1e6
            ho["t1"]=csp_t1/1e6
            ho["lags"]=csp_mlags
#            ho["n_ipp"]=csp_idx
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

#            plt.pcolormesh(csp_acf.real)
 #           plt.colorbar()
            plt.savefig("results/csp_acf-%06d.png"%(csp_t0/1e6))
            plt.close()

            csp_rx_lags[:,:]=0.0
            csp_tx_lags[:,:]=0.0

        elif current_ipp == 10000:
            clp_t1=i0

            # Maximum likelihood estimate of each lag with FFT
            for lag_i in range(len(clp_lags)):
                clp_acf[:,lag_i]=n.fft.ifft((1.0/(clp_tx_lags[:,lag_i]))*clp_rx_lags[:,lag_i])
                
            fname="clp_acf-%06d.h5"%(clp_t0/1e6)
            print("saving %s"%(fname))
            ho=h5py.File("results/clp_acf-%06d.h5"%(clp_t0/1e6),"w")
            ho["t0"]=clp_t0/1e6
            ho["t1"]=clp_t1/1e6
            ho["lags"]=clp_mlags
            ho["n_ipp"]=clp_idx
            ho["ranges"]=n.arange(rmax/rdec)*0.15*rdec
            ho["acf"]=clp_acf
            ho.close()
            # zero stuff
            clp_rx_lags[:,:]=0.0
            clp_tx_lags[:,:]=0.0
            clp_idx=0
            
            norm_acf=n.copy(clp_acf)
            for rg_idx in range(norm_acf.shape[0]):
                norm_acf[rg_idx,:]=norm_acf[rg_idx,:]/norm_acf[rg_idx,0].real
            plt.clf()
            plt.pcolormesh(norm_acf.real,vmin=-1.0,vmax=1.0)
            plt.colorbar()
            plt.savefig("results/nacf-%06d.png"%(clp_t0/1e6))
            plt.close()

        csp_t0=i0
        clp_t0=i0
        current_ipp=ipp
    
    if ipp < 9000:
        # csp, do pulse to pulse lags
        if csp_t0==0:
            csp_t0=i0
        
        # check that we have enough ipps to do all lags
        # notch out stuff at the same time
        n_same_ipps=0
        for pi in range(n_csp_lags-1):
            this_ipp=txs[i+1+pi]-txs[i+pi]
#            print(this_ipp)
            if this_ipp == current_ipp:
                n_same_ipps+=1

        # if we have enough ipps, procede
        if n_same_ipps == (n_csp_lags-1):

            z0=stuffr.decimate(d.read_vector_c81d(i0*25L-25L,rmax*25L,"gr"),dec=25)
            z0_tx=get_tx_vec(z0,current_ipp,rmax)
            z0_rx=get_rx_vec(z0,current_ipp,rmax)            

            for lag_i in range(n_csp_lags-1):
                csp_mlags[lag_i]=lag_i*current_ipp
                z1=stuffr.decimate(d.read_vector_c81d(i0*25L-25L+25L*current_ipp*lag_i,rmax*25L,"gr"),dec=25)
                z1_tx=get_tx_vec(z1,current_ipp,rmax)
                z1_rx=get_rx_vec(z1,current_ipp,rmax)
#                plt.plot(z1_tx.real)
 #               plt.show()
  #              plt.plot(z1_rx.real)
   #             plt.show()
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
                
                    txz=stuffr.decimate(z0_tx*n.conj(z1f_tx),dec=rdec)/1e6
                    rxz=stuffr.decimate(z0_rx*n.conj(z1f_rx),dec=rdec)/1e6

                    rxvar=n.mean(n.abs(rxz))**2.0
#                    rxvar=1.0

#                    plt.plot(txz.real)
 #                   plt.show()
  #                  plt.plot(rxz.real)
   #                 plt.show()                    
                    T=n.fft.fft(txz)
                    R=n.fft.fft(rxz)
                    csp_rx_lags[:,lag_i]+=n.conj(T)*(1.0/rxvar)*R
                    csp_tx_lags[:,lag_i]+=(1.0/rxvar)*n.abs(T)**2.0


    elif ipp == 10000:
        if clp_t0==0:
            clp_t0=i0
        # clp, do normal lags
        z0=stuffr.decimate(d.read_vector_c81d(i0*25L-25L,rmax*25L,"gr"),dec=25)
        print(i0)
        z0_tx=n.copy(z0)
        z0_rx=n.copy(z0)
        z0_tx[460:len(z0_tx)]=0.0 # rx blank
        z0_rx[0:(440+190+1)]=0.0   # tx blank
        clp_idx+=1
        for lag_i in range(len(clp_lags)):
            f_lags=clp_lags[lag_i]
            for flag_i in range(len(f_lags)):
                lag=f_lags[flag_i]
                z1_tx=n.copy(z0_tx)
                z1_rx=n.copy(z0_rx)
                z1_tx=n.roll(z1_tx,lag)
                z1_rx=n.roll(z1_rx,lag)
                
                txz=stuffr.decimate(z0_tx*n.conj(z1_tx),dec=rdec)/1e6
                rxz=stuffr.decimate(z0_rx*n.conj(z1_rx),dec=rdec)/1e6
#                rxvar=n.mean(n.abs(rxz))**2.0
                rxvar=1.0
                T=n.fft.fft(txz)
                R=n.fft.fft(rxz)
                clp_rx_lags[:,lag_i]+=n.conj(T)*(1.0/rxvar)*R
                clp_tx_lags[:,lag_i]+=(1.0/rxvar)*n.abs(T)**2.0

            
        


                
                
        

