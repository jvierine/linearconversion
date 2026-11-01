#!/usr/bin/env python
#
# Four step process to figure out modes:
# 1. find_tx_e.py to detect pulse edges
# 2. find_modes.py to detect ipps
# 3. group_modes.py to group integration periods together
# 4. analyze with mpi paralellization using "intervals.h5" file
# 
# Juha Vierinen
# 
import numpy as n
import matplotlib.pyplot as plt
import sys
sys.path.append("/home/vierinen/lib64/python/")

import digital_rf_hdf5 as drf
import stuffr
import h5py

dname="/media/USRPJUHA/tuesday"
d=drf.read_hdf5(dname)
b=d.get_bounds("gr")
print(b)

i0=b[0]

dec=25
dt=1
sr=25e6
ds=long(dt*sr)
idx=i0

txs=[]
txe=[]
n_block=0
while idx < (b[1]-ds):
    # 25 MHz data
    try:
        z=d.read_vector_c81d(idx,ds,"gr")
    except:
        idx+=ds
    
    # 1 MHz data
    za=stuffr.decimate(n.abs(z),dec=dec)
    tx=n.copy(za)
#    plt.plot(za[0:10000])
 #   plt.show()

    tx[za>1300.0]=1.0
    tx[za<1300.0]=0.0

    txd=n.diff(tx)
    tx_start=n.where(txd == 1.0)[0]+int(idx/25)
    tx_end=n.where(txd == -1.0)[0]+int(idx/25)

    for ts in tx_start:
        txs.append(ts)
    for te in tx_end:
        txe.append(te)
    idx+=ds

    ipp_lens=n.unique(n.diff(tx_start))
    print(ipp_lens)
    print("%d remaining %f (min) ipp %d n_tx %d"%(idx,(b[1]-idx)/25e6/60.0,ipp_lens[0],len(tx_start)))
    
    n_block+=1
    if n_block % 60 == 0:
        ho=h5py.File("%s/tx_ereg.h5"%(dname),"w")
        ho["tx_start"]=txs
        ho["tx_end"]=txe
        ho.close()

ho=h5py.File("%s/tx_ereg.h5"%(dname),"w")
ho["tx_start"]=txs
ho["tx_end"]=txe
ho.close()
