

import numpy as np
import sys,time
import matplotlib.pyplot as plt
from timeit import default_timer
import os.path
import scipy.io

###############################################################################
# functions
def tic():
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    if 'startTime_for_tictoc' in globals():
        print("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " s")
    else:
        print("Toc: start time not set")

################################################################
# configs
################################################################
weight_ux=0.25;
uxF=np.load('Results_uxFNO_t1.npy')
uxS=np.load('Results_uxSimu_t1.npy')
uzF=np.load('Results_uzFNO_t1.npy')
uzS=np.load('Results_uzSimu_t1.npy')
metrics1=np.load('metrics_t1.npy')
cc=0.95
no=1;
cmap='rainbow'
weight_ux=0.15

uxF=uxF/weight_ux;
uxS=uxS/weight_ux;

fig = plt.figure(num=no,figsize=(18, 8)) ;
plt.clf();
#plt.subplot(211)
plt.plot(metrics1[:,1]/metrics1[0,1].max(),'r', label='training data')  #,vmin=-vmax*cc,vmax=cc*vmax);
plt.plot(metrics1[:,2]/metrics1[0,2].max(),'g--', label='test data')#,vmin=-vmax*cc,vmax=cc*vmax);
#plt.semilogy(metrics1[:,1]/metrics1[0,1].max(),'r')  #,vmin=-vmax*cc,vmax=cc*vmax);
#plt.semilogy(metrics1[:,2]/metrics1[0,2].max(),'g--')#,vmin=-vmax*cc,vmax=cc*vmax);
plt.axis([0 ,100,0.2,1])
plt.grid()
plt.title('Misfit Function',fontsize=14);
plt.xlabel('Epochs',fontsize=14);
plt.ylabel('Normalised Misfit',fontsize=14);
plt.legend(fontsize=14)

plt.savefig('test1_metrics.jpg');

#########################################################################
#########################################################################
#########################################################################

# Creating grids
dx=10;
dz=10;

Y,X = np.mgrid[0:600:10, 0:2000:10]
U=uxS.T
V=uzS.T
mark=0;
for i in range(0,60):
    if(uxS[70,i]==0):
        mark=i
        #print(mark)
mark=mark*dz;
x1=70*dx
x2=70.5*dx
       
fig = plt.figure(num=no+1,figsize=(20, 10)) ;
plt.clf();
vmax = np.max(uxS)
vmin = np.min(uxS)
mm1=np.copy(uxS)
mm1[mm1 ==0] = 'nan'
pp=np.copy(mm1);

plt.subplots_adjust( hspace=0.5 )
plt.subplot(321)
plt.pcolormesh(X,Y,mm1.T,vmin=-vmax*cc,vmax=cc*vmax,shading='gouraud',cmap=cmap);
plt.colorbar(label='Velocity (m/s)');
plt.title('Horizontal Wind Speed : Simulation',fontsize=14);
plt.xlabel('Distance (m)',fontsize=14);
plt.ylabel('Height (m)',fontsize=14);
plt.streamplot(X,Y, U,V, density=0.5, color='k', linewidth=1, broken_streamlines=False)
plt.plot([x1,x1],[mark+70,mark+130],'k')
plt.plot([x2,x2],[mark+70,mark+130],'k')
#plt.tight_layout()


mm1=[];
mm1=uxF
mm1[uxS ==0] = 'nan'
qq=np.copy(mm1)
plt.subplot(323)
plt.pcolormesh(X,Y,mm1.T,vmin=-vmax*cc,vmax=cc*vmax,shading='gouraud',cmap=cmap);
plt.colorbar(label='Velocity (m/s)');
plt.streamplot(X,Y, U,V, density=0.5, color='k', linewidth=1, broken_streamlines=False)
plt.plot([x1,x1],[mark+70,mark+130],'k')
plt.plot([x2,x2],[mark+70,mark+130],'k')
plt.title('Horizontal Wind Speed : FNO',fontsize=14);
plt.xlabel('Distance (m)',fontsize=14);
plt.ylabel('Height (m)',fontsize=14);

mm1=[];
mm1=uxS-uxF
mm1[uxS ==0] = 'nan'
plt.subplot(325)
plt.pcolormesh(X,Y,mm1.T,vmin=-vmax*cc,vmax=cc*vmax,shading='gouraud',cmap=cmap);
#plt.streamplot(X,Y, U,V, density=0.5, color='k', linewidth=1, broken_streamlines=False)
plt.plot([x1,x1],[mark+70,mark+130],'k')
plt.plot([x2,x2],[mark+70,mark+130],'k')
plt.colorbar(label='Velocity (m/s)');
plt.title('Difference Horizontal Wind Speed',fontsize=14);
plt.xlabel('Distance (m)',fontsize=14);
plt.ylabel('Height (m)',fontsize=14);

vmax = np.max(uzS)
vmin = np.min(uzS)
mm1=np.copy(uzS)
mm1[mm1 ==0] = 'nan'
plt.subplot(322)
plt.pcolormesh(X,Y,mm1.T,vmin=-vmax*cc,vmax=cc*vmax,shading='gouraud',cmap=cmap);
plt.colorbar(label='Velocity (m/s)');
plt.streamplot(X,Y, U,V, density=0.5, color='k', linewidth=1, broken_streamlines=False)
plt.plot([x1,x1],[mark+70,mark+130],'k')
plt.plot([x2,x2],[mark+70,mark+130],'k')
plt.title('Vertical Wind Speed : Simulation',fontsize=14);
plt.xlabel('Distance (m)',fontsize=14);
plt.ylabel('Height (m)',fontsize=14);

mm1=[];
mm1=uzF
mm1[uzS ==0] = 'nan'
plt.subplot(324)
plt.pcolormesh(X,Y,mm1.T,vmin=-vmax*cc,vmax=cc*vmax,shading='gouraud',cmap=cmap);
plt.colorbar(label='Velocity (m/s)');
plt.streamplot(X,Y, U,V, density=0.5, color='k', linewidth=1, broken_streamlines=False)
plt.plot([x1,x1],[mark+70,mark+130],'k')
plt.plot([x2,x2],[mark+70,mark+130],'k')
plt.title('Vertical Wind Speed : FNO',fontsize=14);
plt.xlabel('Distance (m)',fontsize=14);
plt.ylabel('Height (m)',fontsize=14);

mm1=[];
mm1=uzS-uzF
mm1[uzS ==0] = 'nan'
plt.subplot(326)
plt.pcolormesh(X,Y,mm1.T,vmin=-vmax*cc,vmax=cc*vmax,shading='gouraud',cmap=cmap);
plt.colorbar(label='Velocity (m/s)');
#plt.streamplot(X,Y, U,V, density=0.5, color='k', linewidth=1, broken_streamlines=False)
plt.plot([x1,x1],[mark+70,mark+130],'k')
plt.plot([x2,x2],[mark+70,mark+130],'k')
plt.title('Difference Vertical Wind Speed',fontsize=14);
plt.xlabel('Distance (m)',fontsize=14);
plt.ylabel('Height (m)',fontsize=14);

plt.savefig('test1_snaps.jpg');

#########################################################################
#########################################################################
#########################################################################


fig = plt.figure(num=no+2,figsize=(10, 20)) ;
plt.clf();
plt.subplots_adjust( wspace=0.5 )
plt.subplot(131)
i=50;
plt.plot(pp[i,:],Y[:,i],'r',linewidth=2)
#plt.plot(qq[i,:],Y[:,i],'r--',linewidth=2)
plt.axis([0,20,0,600])
plt.grid()
plt.text(1,580,'(A)',fontsize=14);
plt.xlabel('Velocity (m/s)',fontsize=14);
plt.ylabel('Height (m)',fontsize=14);
plt.legend(fontsize=14)

plt.subplot(132)
i=73
plt.plot(pp[i,:],Y[:,i],'b',linewidth=2)
#plt.plot(qq[i,:],Y[:,i],'b--',linewidth=2)
plt.axis([0,20,0,600])
plt.grid()
plt.text(1,580,'(B)',fontsize=14);
plt.xlabel('Velocity (m/s)',fontsize=14);
plt.ylabel('Height (m)',fontsize=14);
plt.legend(fontsize=14)

plt.subplot(133)
i=150
plt.plot(pp[i,:],Y[:,i],'g',linewidth=2)
#plt.plot(qq[i,:],Y[:,i],'g--',linewidth=2)
plt.axis([0,20,0,600])
plt.grid()
plt.text(1,580,'(C)',fontsize=14);
plt.xlabel('Velocity (m/s)',fontsize=14);
plt.ylabel('Height (m)',fontsize=14);
plt.legend(fontsize=14)

plt.savefig('test1_profile.jpg');

