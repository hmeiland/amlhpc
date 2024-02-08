# %%

###########################################################################
###########################################################################
# load libaries
import time
import random
import numpy as np
import pandas as pd
import dask
import zarr
import os
import sys

from dask.distributed import wait
import pylops
from pylops.utils.wavelets import ricker
import matplotlib . pyplot as plt
###########################################################################
###########################################################################


# %%
##########################################################################
# parameters
#ZARR blob account - testing
AZURE_STORAGE_ACCOUNT_NAME="blobtest12345"
AZURE_STORAGE_ACCESS_KEY="xxxx"
container_name="seismic"

backend='zarr_blob'
object_name1='ZarrVolume0'
object_name2='ZarrVolume1'

dh=100;
dx=12.5
dt=0.004;

print('done this cell')

# %%
#print('Number of arguments:', len(sys.argv), 'arguments.')
print('Argument List:', str(sys.argv) )

nprocess = int(sys.argv[1])
job_id = int(sys.argv[2])

#nprocess = 8
#job_id = 4


# %%
###########################################################################
###########################################################################
#define functions
def filter_gather(cig):
    
    # adjoint Radon transform
    xadj = Rop.H * cig
    # filtering on adjoint transform
    xfilt = np.zeros_like(xadj)
    xfilt[npx // 2 - 3 : npx // 2 + 4] = xadj[npx // 2 - 3 : npx // 2 + 4]
    cig_filt = Rop * xfilt

    return cig_filt

#read write fucntion
def task_to_be_done(il1,il2):
    
    il1 = int(il1)
    il2 = int(il2) + 1
    # This is for geophysucs
    for il in range(il1,il2):
        print('writing out inline',il)
        for xl in range(0,nxlines):
            # physics functions
            izarrO.volume[il,xl,:,:]= filter_gather(izarrI.volume[il,xl,:,:])
            izarrO.header[il,xl,:,:]= izarrI.header[il,xl,:,:]
            #for o in range(0,noffsets):
            #    hea[il,xl,o,0] = il
            #    hea[il,xl,o,1] = xl
            #    hea[il,xl,o,2] = o

    #izarrO.header = izarrI.header

    return il1,il2,noffsets

###########################################################################

# functions to measure time intervals
def tic():
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    if 'startTime_for_tictoc' in globals():
        print("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " s")
    else:
        print("Toc: start time not set")
        
###########################################################################
###########################################################################


# %%
###########################################################################
###########################################################################
# opening zarr input seismic file
tic()
if backend == 'zarr_blob':       
    import azure.storage.blob 
    print('opening input file on blob to process',object_name1)
    
    container_client = azure.storage.blob.ContainerClient(
        account_name=AZURE_STORAGE_ACCOUNT_NAME,
        credential=AZURE_STORAGE_ACCESS_KEY,
        account_url="https://{}.blob.core.windows.net".format(AZURE_STORAGE_ACCOUNT_NAME),
        container_name=container_name)
    store = zarr.ABSStore(client=container_client, prefix=object_name1)
    root = zarr.group(store=store, overwrite=False)
    izarrI=root

elif backend == 'zarr':
    print('opening input file to process',object_name1)
    izarrI = zarr.open(object_name1, mode='r')
toc()
#print(izarr.volume[1,1,1,0:3])
#print(izarr.header[1,1,1,0:3])

nilines, nxlines,noffsets,nsamples = izarrI.volume.shape
nilines, nxlines,noffsets,nheaders = izarrI.header.shape
ntasks=nilines
print('nil=',nilines,' nxl=',nxlines,' noff=',noffsets,' ntasks=',ntasks)
     
###########################################################################
###########################################################################

# %%
###########################################################################
###########################################################################
# opening zarr output file
shape_v = (nilines,nxlines,noffsets,nsamples)
shape_h = (nilines,nxlines,noffsets,nheaders)
# parallel partition in inlines so need to chunk over this
chunk_v = (1,nxlines,noffsets,nsamples)
chunk_h = (1,nxlines,noffsets,nheaders)

if backend == 'zarr':
    print('opening input file to process',object_name2)
    izarrO = zarr.open(object_name2, mode='r')

elif backend == 'zarr_blob':
    import azure.storage.blob
    container_client = azure.storage.blob.ContainerClient(
        account_name=AZURE_STORAGE_ACCOUNT_NAME,
        credential=AZURE_STORAGE_ACCESS_KEY,
        account_url="https://{}.blob.core.windows.net".format(AZURE_STORAGE_ACCOUNT_NAME),
        container_name=container_name)
    store = zarr.ABSStore(client=container_client, prefix=object_name2)
    root = zarr.group(store=store, overwrite=False)
    izarrO = root
    #vol = root.zeros('volume', chunks=chunk_v, shape=shape_v, dtype=np.float32)
    #hea = root.zeros('header', chunks=chunk_h, shape=shape_h, dtype=np.float32)
    
###########################################################################
###########################################################################


# %%
###########################################################################
###########################################################################
# define radon filter
par = {"ox": 0, "dx": dh, "nx": noffsets, "ot": 0, "dt": dt, "nt": nsamples, "f0": 30}

# create axis
taxis, taxis2, xaxis, yaxis = pylops.utils.seismicevents.makeaxis(par)

# radon operator
npx = 51
pxmax = 5e-5  # s/m
px = np.linspace(-pxmax, pxmax, npx)

Rop = pylops.signalprocessing.Radon2D(
    taxis, xaxis, px, kind="linear", interp="nearest", centeredh=False, dtype="float64")

###########################################################################
###########################################################################


# %%
###########################################################################
# processsing gathers and writing out
print('processing seismic file of size ',shape_v)
print('writing out...')

i=job_id
il1 = np.floor(i*(nilines/nprocess)) 
il2 = np.floor((i+1)*(nilines/nprocess))-1

# serial version
tic()
print(il1,il2)
task_to_be_done(il1,il2)
toc()

##########################################################################
##########################################################################
# %%
##########################################################################
##########################################################################
figures_on=0
no=12
slx=int(nilines/2);
sly=int(nxlines/2)
slz=int(nsamples/2)

#%matplotlib inline

if(figures_on==1):  
        # opening zarr input seismic file
        #tic()
        # opening zarr input seismic file
        print('opening input file to process',object_name1)
        if backend == 'zarr_blob':            
            container_client = azure.storage.blob.ContainerClient(
                account_name=AZURE_STORAGE_ACCOUNT_NAME,
                credential=AZURE_STORAGE_ACCESS_KEY,
                account_url="https://{}.blob.core.windows.net".format(AZURE_STORAGE_ACCOUNT_NAME),
                container_name=container_name)
            store = zarr.ABSStore(client=container_client, prefix=object_name1)
            root = zarr.group(store=store, overwrite=False)
            izarr=root
            
        elif backend == 'zarr':
            print('opening input file to process',object_name1)
            izarr = zarr.open(object_name1, mode='r')
        #toc()
        
        fig = plt.figure(num=no,figsize=(9, 6)) ;
        vv = 0.02*np.max(izarr.volume[:,:,:,:]);
        plt.clf();
        
        mm = izarr.volume[slx,:,0,:]
        plt.subplot(221)
        plt.imshow(np.transpose(mm),cmap='seismic',extent=(0,nxlines,nsamples*dt,0))#,vmin=-vv,vmax=vv,cmap='seismic' );#, cmap="RdGy")
        #plt.colorbar();
        plt.gca().axis('tight')
        plt.ylabel('Time (s)',fontsize=12)
        plt.xlabel('xline',fontsize=12)
        #plt.gca().invert_yaxis()  
        
        mm = izarr.volume[:,sly,0,:]
        plt.subplot(222)
        plt.imshow(np.transpose(mm),extent=(0,nilines,nsamples*dt,0),cmap='seismic' );#, cmap="RdGy")
        #plt.colorbar();        
        plt.gca().axis('tight')
        plt.ylabel('Time (s)',fontsize=12)
        plt.xlabel('inline',fontsize=12)
        #plt.gca().invert_yaxis()
        
        mm=[];
        mm = izarr.volume[:,:,2,slz]
        plt.subplot(223)
        plt.imshow(mm,cmap='seismic');#, cmap="RdGy")
        #plt.gca().invert_yaxis()
        plt.gca().axis('tight')
        plt.ylabel('inlines',fontsize=12)
        plt.xlabel('xlines',fontsize=12)
        #plt.colorbar();

        mm=[];
        mm = izarr.volume[slx,sly,:,:]
        plt.subplot(224)
        plt.imshow(np.transpose(mm),cmap='seismic',extent=(0,noffsets*dh,nsamples*dt,0) )#,vmin=-vv,vmax=vv,cmap='seismic');#, cmap="RdGy")
        #plt.gca().invert_yaxis()
        plt.gca().axis('tight')
        plt.ylabel('Time (s)',fontsize=12)
        plt.xlabel('Offset (m)',fontsize=12)
        
        # opening zarr ouput seismic file
        print('opening input file to process',object_name2)
        if backend == 'zarr_blob':            
            container_client = azure.storage.blob.ContainerClient(
                account_name=AZURE_STORAGE_ACCOUNT_NAME,
                credential=AZURE_STORAGE_ACCESS_KEY,
                account_url="https://{}.blob.core.windows.net".format(AZURE_STORAGE_ACCOUNT_NAME),
                container_name=container_name)
            store = zarr.ABSStore(client=container_client, prefix=object_name2)
            root = zarr.group(store=store, overwrite=False)
            izarr2=root
            
        elif backend == 'zarr':
            print('opening input file to process',object_name2)
            izarr2 = zarr.open(object_name2, mode='r')
        
        fig = plt.figure(num=no+1,figsize=(9, 6)) ;
        vv = 0.02*np.max(izarr2.volume[:,:,:,:]);
        plt.clf();
        
        mm = izarr2.volume[slx,:,0,:]
        plt.subplot(221)
        plt.imshow(np.transpose(mm),cmap='seismic',extent=(0,nxlines,nsamples*dt,0))#,vmin=-vv,vmax=vv,cmap='seismic' );#, cmap="RdGy")
        #plt.colorbar();
        plt.gca().axis('tight')
        plt.ylabel('Time (s)',fontsize=12)
        plt.xlabel('xline',fontsize=12)
        #plt.gca().invert_yaxis()  
        
        mm = izarr2.volume[:,sly,0,:]
        plt.subplot(222)
        plt.imshow(np.transpose(mm),extent=(0,nilines,nsamples*dt,0),cmap='seismic' );#, cmap="RdGy")
        #plt.colorbar();        
        plt.gca().axis('tight')
        plt.ylabel('Time (s)',fontsize=12)
        plt.xlabel('inline',fontsize=12)
        #plt.gca().invert_yaxis()
        
        mm=[];
        mm = izarr2.volume[:,:,2,slz]
        plt.subplot(223)
        plt.imshow(mm,cmap='seismic');#, cmap="RdGy")
        #plt.gca().invert_yaxis()
        plt.gca().axis('tight')
        plt.ylabel('inlines',fontsize=12)
        plt.xlabel('xlines',fontsize=12)
        #plt.colorbar();

        mm=[];
        mm = izarr2.volume[slx,sly,:,:]
        plt.subplot(224)
        plt.imshow(np.transpose(mm),cmap='seismic',extent=(0,noffsets*dh,nsamples*dt,0) )#,vmin=-vv,vmax=vv,cmap='seismic');#, cmap="RdGy")
        #plt.gca().invert_yaxis()
        plt.gca().axis('tight')
        plt.ylabel('Time (s)',fontsize=12)
        plt.xlabel('Offset (m)',fontsize=12)
        
        ss="radon_file_py"+str(job_id)+".jpeg"
        plt.savefig(ss,dpi=300);   
###########################################################################
###########################################################################


# %%



