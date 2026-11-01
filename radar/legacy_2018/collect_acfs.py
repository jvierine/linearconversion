#!/usr/bin/env

import numpy as n
import matplotlib.pyplot as plt
import h5py
import glob

h=h5py.File("results3/clp_acf-1525452405.h5","r")
lags_clp=n.copy(h["lags"].value)
h.close()
lags_csp=n.array([0,40,80,1000,1250,1500,1750,2000,2250,2500,2750,3000,3250,3500],dtype=n.int)

fl=glob.glob("results3/c*.h5")
fl.sort()
nf=len(fl)
S=n.zeros([nf,5000],dtype=n.complex)
tv=n.zeros(nf)
rv=n.zeros(5000)
tv2=n.arange(nf)

dt=1800

for fi,f in enumerate(fl):
    print(f)
    h=h5py.File(f,"r")
    flags=h["lags"].value[0:4]
    for i in range(len(lags)):
        idx=n.where(flags[i] == lags_csp)[0]
        print("%d %1.2f"%(idx,lags_csp[i]))
    tv[fi]=h["t0"].value
    h.close()
idx=n.argsort(tv)
print(idx)
print(tv[idx])

#S=n.zeros([5000,len(

#for fi,f in enumerate(fl):
 #   print(f)
  #  h=h5py.File(f,"r")
  #  uts=n.copy(h["t0"].value)
  #  flags=n.copy(h["lags"].value[0:4])
    
#    S[fi,:]=h["acf"].value[:,1]/n.real(h["acf"].value[:,0])
 #   tv[fi]=h["t0"].value
#    h.close()
