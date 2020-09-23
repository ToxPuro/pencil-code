# get_stats.py
#
# 05-may-20
# Author: F. Gent (fred.gent.ncl@gmail.com).
#
""" Derive auxilliary data and other diagnostics from var.h5 file and 
    save to new h5 file
 
    uses:
      compute 'data' arrays of size [nz,ny,nx] as required
      store 'time' of snapshot
      compute 'masks' for example by temperature phase
      compute summary statistics 'stats' 
      compute 'structure' functions as required
"""
import numpy as np
import scipy
from . import is_vector
from scipy.interpolate import interp1d
from ..math import dot, dot2, natural_sort, helmholtz_fft, cpu_optimal
from ..math.derivatives import curl, div, curl2, grad
from ..calc import fluid_reynolds, magnetic_reynolds
from ..io import open_h5, group_h5, dataset_h5
from fileinput import input
from sys import stdout
import subprocess as sub
from .. import read 
import os

def derive_stats(sim_path, src, dst, stat_keys=['Rm', 'uu', 'Ms'], par=[],
                 comm=None, overwrite=False, rank=0, size=1, nghost=3,
                 status='a', chunksize = 1000.0, quiet=True, nmin=32,
                 lmask=False, mask_key = 'hot'
                ):

    if comm:
        overwrite = False
    if isinstance(par, list):
        os.chdir(sim_path)
        par = read.param(quiet=True,conflicts_quiet=True)
    #get data dimensions
    nx, ny, nz = src['settings']['nx'][0],\
                 src['settings']['ny'][0],\
                 src['settings']['nz'][0]
    mx, my, mz = src['settings']['mx'][0],\
                 src['settings']['my'][0],\
                 src['settings']['mz'][0]
    #split data into manageable memory chunks
    dstchunksize = 8*nx*ny*nz/1024*1024
    if dstchunksize > chunksize:
        nchunks = cpu_optimal(nx,ny,nz,quiet=quiet,
                              mvar=src['settings/mvar'][0],
                              maux=src['settings/maux'][0],
                              MBmin=chunksize,nmin=nmin,size=size)[1]
    else:
        nchunks = [1,1,1]
    print('nchunks {}'.format(nchunks)) 
    allchunks = nchunks[0]*nchunks[1]*nchunks[2]
    # for mpi split chunks across processes
    if size > 1:
        #allchunks = np.arange(nchunks[0]*nchunks[1]*nchunks[2])
        locindx = np.array_split(np.arange(nx)+nghost,nchunks[0]) 
        locindy = np.array_split(np.arange(ny)+nghost,nchunks[1]) 
        locindz = np.array_split(np.arange(nz)+nghost,nchunks[2])
        #ichunks = np.array_split(allchunks,size)
        indx = [locindx[np.mod(rank,nchunks[0])]]
        indy = [locindy[np.mod(rank,nchunks[1])]]
        indz = [locindz[np.mod(rank,nchunks[2])]]
        if rank >= nchunks[2]:
            indy = [locindy[np.mod(rank+1,nchunks[1])]]
        if rank >= nchunks[1]:
            indx = [locindx[np.mod(rank+1,nchunks[0])]]
            if rank >= nchunks[2]:
                indx = [locindx[np.mod(rank+2,nchunks[0])]]
    else:
        locindx = np.array_split(np.arange(nx)+nghost,nchunks[0]) 
        locindy = np.array_split(np.arange(ny)+nghost,nchunks[1]) 
        locindz = np.array_split(np.arange(nz)+nghost,nchunks[2])
        indx = np.array_split(np.arange(nx)+nghost,nchunks[0]) 
        indy = np.array_split(np.arange(ny)+nghost,nchunks[1]) 
        indz = np.array_split(np.arange(nz)+nghost,nchunks[2])
    # ensure derived variables are in a list
    if isinstance(stat_keys, list):
        stat_keys = stat_keys
    else:
        stat_keys = [stat_keys]
    # initialise group 
    group = group_h5(dst, 'stats', status='a', overwrite=overwrite,
                     comm=comm, rank=rank, size=size)
    for key in stat_keys:
        mean_stat = list()
        stdv_stat = list()
        mean_mask = list()
        stdv_mask = list()
        nmask_msk = list()
        mean_nmsk = list()
        stdv_nmsk = list()
        nmask_nmk = list()
        for iz in indz:
            n1, n2 = iz[ 0],\
                     iz[-1]+1
            for iy in indy:
                m1, m2 = iy[ 0],\
                         iy[-1]+1
                for ix in indx:
                    l1, l2 = ix[ 0],\
                             ix[-1]+1
                    if key in src['data'].keys():
                        var = src['data'][key][n1:n2,m1:m2,l1:l2]
                    elif key == 'uu' or key == 'aa':
                        tmp = np.array(
                              [src['data'][key[0]+'x'][n1:n2,m1:m2,l1:l2],
                               src['data'][key[0]+'y'][n1:n2,m1:m2,l1:l2],
                               src['data'][key[0]+'z'][n1:n2,m1:m2,l1:l2]])
                        var = np.sqrt(dot2(tmp))
                    else:
                        if key in dst['data'].keys():
                            if is_vector(key):
                                var = np.sqrt(dot2(
                                    dst['data'][key][:,n1:n2,m1:m2,l1:l2]))
                            else:
                                var = dst['data'][key][n1:n2,m1:m2,l1:l2]
                        else:
                            print('stats: '+key+' does not exist in ',
                                    src,'or',dst)
                            continue
                    if lmask:
                        mask = dst['masks'][mask_key][0,n1:n2,m1:m2,l1:l2]
                        Nmask = mask[mask==False].size
                        if Nmask > 0:
                            mean_mask.append(var[mask==False].mean()*Nmask) 
                            stdv_mask.append(var[mask==False].std()*Nmask)
                        else: 
                            mean_mask.append(0) 
                            stdv_mask.append(0)
                        nmask_msk.append(Nmask)
                        nmask = mask[mask==True].size
                        if nmask > 0:
                            mean_nmsk.append(var[mask==True].mean()*nmask)
                            stdv_nmsk.append(var[mask==True].std()*nmask)
                        else: 
                            mean_nmsk.append(0)
                            stdv_nmsk.append(0)
                        nmask_nmk.append(nmask) 
                    mean_stat.append(var.mean())
                    stdv_stat.append(var.std())
        if comm:
            if lmask:
                comm.gather(mean_mask, root=0)
                comm.gather(stdv_mask, root=0)
                comm.bcast(mean_mask, root=0)
                comm.bcast(stdv_mask, root=0)
                comm.gather(mean_nmsk, root=0)
                comm.gather(stdv_nmsk, root=0)
                comm.bcast(mean_nmsk, root=0)
                comm.bcast(stdv_nmsk, root=0)
                comm.gather(nmask_msk, root=0)
                comm.gather(nmask_nmk, root=0)
                comm.bcast(nmask_msk, root=0)
                comm.bcast(nmask_nmk, root=0)
            comm.gather(mean_stat, root=0)
            comm.gather(stdv_stat, root=0)
            comm.bcast(mean_stat, root=0)
            comm.bcast(stdv_stat, root=0)
        if lmask:
            summk = np.sum(nmask_msk)
            if summk > 0: 
                meanm = np.sum(mean_mask)/summk
                stdvm = np.sum(stdv_mask)/summk
            else:
                meanm = 0
                stdvm = 0
            sumnk = np.sum(nmask_nmk)
            if sumnk > 0: 
                meann = np.sum(mean_nmsk)/sumnk
                stdvn = np.sum(stdv_nmsk)/sumnk
            else:
                meann = 0
                stdvn = 0
            print(mask_key+'-'+key+'-mean = {}, '.format(meanm)+
                  mask_key+'-'+key+'-std = {}'.format(stdvm))
            print('not-'+mask_key+'-'+key+'-mean = {}, '.format(meann)+
                  'not-'+mask_key+'-'+key+'-std = {}'.format(stdvn))
            dataset_h5(group, mask_key+'-'+key+'-mean', status=status,
                       data=meanm, comm=comm, size=size, rank=rank,
                       overwrite=overwrite)
            dataset_h5(group, mask_key+'-'+key+'-std', status=status,
                       data=stdvm, comm=comm, size=size, rank=rank,
                       overwrite=overwrite)
            dataset_h5(group, 'not-'+mask_key+'-'+key+'-mean', status=status,
                       data=meann, comm=comm, size=size, rank=rank,
                       overwrite=overwrite)
            dataset_h5(group, 'not-'+mask_key+'-'+key+'-std', status=status,
                       data=stdvn, comm=comm, size=size, rank=rank,
                       overwrite=overwrite)
        mstat = np.mean(mean_stat)
        dstat = np.mean(stdv_stat)
        print(key+'-mean = {}, '.format(mstat)+key+'-std = {}'.format(dstat))
        dataset_h5(group, key+'-mean', status=status, data=mstat,
                   comm=comm, size=size, rank=rank,
                   overwrite=overwrite)
        dataset_h5(group, key+'-std', status=status, data=dstat,
                   comm=comm, size=size, rank=rank,
                   overwrite=overwrite)
