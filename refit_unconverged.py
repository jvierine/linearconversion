"""Continue only iteration-limited native fits, preserving the original rows."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import argparse,hashlib,multiprocessing as mp
from pathlib import Path
import numpy as np,h5py
import fit_full_altitude as fit

def objective(row,i,j):
    mode='csp' if fit.D['z'][j]<94.5 else 'clp'
    n=len(fit.D[mode+'_lags'])
    return (row[8]*(2*n-4)
            +(np.log(row[2]/fit.D['ti0'][i,j])/fit.D['sigma_ti'][j])**2
            +(np.log(row[1]/fit.D['te0'][i,j])/fit.D['sigma_te'][j])**2
            +(row[3]/300.)**2)

def job(task):
    i,j,old=task
    new=fit.cell(i,j,max_nfev=200,start=old[:4])
    old_cost=objective(old,i,j);new_cost=objective(new,i,j)
    accept=new_cost<=old_cost*(1+1e-5)+1e-8
    return i,j,new if accept else old,old_cost,new_cost,accept

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--workers',type=int,default=36);a=p.parse_args()
    path=a.root/'full_altitude_fits.h5'
    with h5py.File(path) as h:
        assert h.attrs['completed']
        results=h['results'][:];z=h['height_km'][:]
    bad=np.argwhere(results[:,:,9]==0)
    print('Iteration-limited cells:',len(bad),flush=True)
    if not len(bad):return
    fit.prepare(a.root)
    offset=int(np.argmin(abs(fit.D['z']-z[0])))
    tasks=[(int(i),int(j)+offset,results[i,j]) for i,j in bad]
    updates=[]
    with mp.get_context('fork').Pool(a.workers,initializer=fit.init) as pool:
        for k,result in enumerate(pool.imap_unordered(job,tasks,chunksize=1)):
            updates.append(result)
            if k%1000==0:print('refitted',k+1,'of',len(tasks),flush=True)
    with h5py.File(path,'r+') as h:
        g=h.create_group('iteration_limit_refit')
        g['indices']=bad;g['original_results']=results[bad[:,0],bad[:,1]]
        costs=[]
        for i,j,new,old_cost,new_cost,accept in updates:
            outj=j-offset;results[i,outj]=new
            costs.append([i,outj,old_cost,new_cost,accept])
        g['objective_comparison']=np.array(costs)
        h['results'][:]=results
        g.attrs['objective_comparison_columns']='time_index,altitude_index,old_objective,new_objective,accepted'
        g.attrs['max_additional_evaluations']=200
        g.attrs['provenance']='refit_unconverged.py; '+hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        g.attrs['cell_source_sha256']=hashlib.sha256(Path(fit.__file__).read_bytes()).hexdigest()
        print('Remaining unconverged:',int(np.count_nonzero(results[:,:,9]==0)),flush=True)

if __name__=='__main__':main()
