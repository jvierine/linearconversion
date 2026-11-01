#!/usr/bin/env python
#
# 1. first run find_tx_e.py to flag
#    potential tx pulses
# 2. then run find_modes to filter valid ipps
# 3. then run group ipps to figure out integration periods
# 
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

dname="/media/USRPJUHA/tuesday"
h=h5py.File("%s/tx_ereg.h5"%(dname),"r")

txs=n.copy(h["tx_start"].value)
txe=n.copy(h["tx_end"].value)

current_ipp=txs[1]-txs[0]    
current_ipp=10*int(current_ipp/10.0)
same_ipps=[]
valid_ipps=n.array([1000,1250,1500,1750,2250,2750,3250,10000])

ipp_intervals=[]

tx_starts={}
for i in valid_ipps:
    tx_starts[i]=[]

for i in range(len(txs)-10):
    ipp=txs[i+1]-txs[i]
    if n.min(n.abs(valid_ipps-ipp)) < 4:
        this_ipp=valid_ipps[n.argmin(n.abs(valid_ipps-ipp))]
        tx_starts[this_ipp].append(txs[i])
        if i%10000==0:
            print("ipp %d n %d"%(this_ipp,len(tx_starts[this_ipp])))
            ho=h5py.File("%s/tx_pulses.h5"%(dname),"w")
            for vi in valid_ipps:
                ho["%d"%(vi)]=n.copy(n.array(tx_starts[vi],dtype=n.uint64))
            ho.close()

ho=h5py.File("%s/tx_pulses.h5"%(dname),"w")
for i in valid_ipps:
    ho["%d"%(i)]=n.copy(n.array(tx_starts[i],dtype=n.uint64))
ho.close()


    
