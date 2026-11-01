#!/usr/bin/env python

import glob
import numpy as n
import matplotlib.pyplot as plt
import h5py
import stuffr


dname="results4"

fl=glob.glob("%s/ext-*.h5"%(dname))
fl.sort()
nf=len(fl)
P=n.zeros([nf,400])
F=n.zeros([nf,400])
tv=n.zeros(nf)
FD=n.zeros([nf,400])

for fi,f in enumerate(fl):
    print(f)
    h=h5py.File(f,"r")
    print(h.keys())
    P[fi,:]=h["P"].value
    F[fi,:]=h["F"].value
    tv[fi]=h["t0"].value
    print(h["mode"].value)
    h.close()

plt.pcolormesh(10.0*n.log10(n.transpose(P)),vmin=0,vmax=30)
plt.colorbar()
plt.show()
plt.pcolormesh(n.transpose(F))
plt.colorbar()
plt.show()

rv=n.arange(400)*0.3
for ri in range(400):
    lpf=n.convolve(n.repeat(1.0/20.0,20),F[:,ri],mode="same")
#    if ri > 300:
 #       plt.plot(lpf-F[:,ri])
  #      plt.show()
    FD[:,ri]=F[:,ri]-lpf
plt.pcolormesh(tv,rv,n.transpose(FD),vmin=-0.05,vmax=0.05)
plt.colorbar()
plt.show()


