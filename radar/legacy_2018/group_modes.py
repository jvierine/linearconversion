#!/usr/bin/env python

import numpy as n
import h5py 

dname="/media/USRPJUHA/tuesday"
h=h5py.File("%s/tx_pulses.h5"%(dname),"r")
print(h.keys())

ipps=n.sort(n.array(h.keys(),dtype=n.int))
print(ipps)

ipp_intervals=[]
#ipp_intervals.append({"ipp":current_ipp,"n_ipp":len(same_ipps),"ipps":same_ipps})

ipp_len=[]
ipp_txs=[]
ipp_nipp=[]

txss={}
for i in ipps:
    txss[i]=n.copy(h["%d"%(i)].value)

for i in ipps:
    txs=txss[i]
    this_txs=[]
    mean_ipp=0.0
    for j in range(len(txs)-1):
        dt=txs[j+1]-txs[j]
        if n.abs( (txs[j+1]-txs[j]) - i ) < 10.0:
            this_txs.append(txs[j])
            mean_ipp+=dt
        else:
            print("bad ipp %d %d"%(dt,txs[j]))
        if dt > 1e6:
            if len(this_txs)>0:
                ipp_intervals.append({"ipp":i,"n_ipp":len(this_txs),"ipps":this_txs})
                ipp_len.append(i)
                ipp_txs.append(this_txs[0])
                ipp_nipp.append(len(this_txs))

                mean_ipp=mean_ipp/float(len(this_txs))
                print("mode %d len %d dt next %1.2f mean_ipp %1.2f"%(i,len(this_txs),dt,mean_ipp))
                this_txs=[]

#    ipp_intervals.append({"ipp":current_ipp,"n_ipp":len(same_ipps),"ipps":same_ipps})
        

h.close()

ho=h5py.File("intervals.h5","w")
ho["ipp"]=n.array(ipp_len,dtype=n.int)
ho["txs"]=n.array(ipp_txs,dtype=n.int)
ho["nipp"]=n.array(ipp_nipp,dtype=n.int)
ho.close()


