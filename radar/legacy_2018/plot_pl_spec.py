#!/usr/bin/env python

import h5py
import matplotlib.pyplot as plt
import numpy as n

h=h5py.File("results4/cpl-1525452324.h5","r")
#plt.pcolormesh(h["PL"].value[0:8192,:],vmax=1e18)
#plt.colorbar()
#plt.show()

plt.plot(10.0*n.log10(h["PL"].value[257,]),label="257")
plt.plot(10.0*n.log10(h["PL"].value[258,]),label="258")
plt.plot(10.0*n.log10(h["PL"].value[259,]),label="259")
plt.plot(10.0*n.log10(h["PL"].value[260,]),label="260")
plt.plot(10.0*n.log10(h["PL"].value[261,]),label="261")
plt.legend()
plt.show()
print(h.keys())
h.close()
