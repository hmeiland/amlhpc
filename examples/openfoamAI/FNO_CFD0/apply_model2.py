

import numpy as np
import sys,time
import matplotlib.pyplot as plt
from timeit import default_timer
import os.path
import scipy.io

sys.path.insert(0, '/mnt/c/Users/gaobrien/Documents/IrelandTurbine2/FNO_CFD0')
from utilities3 import *
from FNO_fn1 import *

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
# inputs and outputs
################################################################

no=10;

filename0 = '../Results/model980.npy';
filename1 = '../Results/ux980.npy';  
filename2 = '../Results/uz980.npy';  
outputimage='inference_snap.jpg'
modelname = 'model/model_t1'
y_normalizer_name = 'ynormal1'

################################################################
################################################################
################################################################
################################################################
# 


NxU = 200 ; # output
NzU = 60;
NxM = 200;  # input model
NzM = 60;
Nd0 = 1; # number model input dimensions
Nd1 = 1;  # number of temporal snapshots
cc=0.95

run=0;
resampX=1;
resampZ=1;
resampXm=1;
resampZm=1;

ntest = 1
ntrain=0
stest=1
skip= 0 ;
weight_ux=0.15

modes = 2
width = 24

batch_size = 1 #  used ~ 20
batch_size2 = batch_size

epochs = 1
learning_rate = 0.005 # 0.001
scheduler_step = 40
scheduler_gamma = 0.5



runtime = np.zeros(2, )
t1 = default_timer()

Nd2=2#int(Nd1/2)

##########################################################################
# load data

cc=0.95
U=( np.zeros((Nd2,NxU,NzU),dtype='float32') )
M=( np.zeros((Nd0,NxM,NzM),dtype='float32') )

T=Nd2
nx=int(NxM/resampXm)+0
nz=int(NzM/resampZm)+0

Nx=int(NxU/resampX)-0
Nz=int(NzU/resampZ)+0


test_a=torch.tensor( np.zeros((ntest, Nd0,nx,nz),dtype='float32') ) 
test_u=torch.tensor( np.zeros((ntest, Nd2,Nx,Nz),dtype='float32') )
count = 0

print(test_a.shape)
print(test_u.shape)

Umax=( np.zeros((ntrain+ntest,3),dtype='float32') )

###############################################################################
tic()
print('Loading data ...')

for lt in range(stest,stest+1):
   
    if(lt%200==0):
        print(lt)
        
    ff = filename0
    check_file = os.path.isfile(ff)
    if(check_file==True):
        model = np.load(ff)

        shapeM=(1,NxM,NzM)
        M[0,:,:] = model
        
        ff = filename1
        ux = np.load(ff)
        ff = filename2
        uz = np.load(ff)
        
        shape=(Nd0,NxU,NzU)
        U[0,:,:] = ux[:,:]*weight_ux
        U[1,:,:] = uz[:,:]
    else:
        print('    No file',ff)
        
    test_a[count,:,:nx,:]=torch.tensor( M[:,::resampXm,::resampZm] )
    test_u[count,0,:,:]=torch.tensor( U[0,::resampX,::resampZ] )
    test_u[count,1,:,:]=torch.tensor( U[1,::resampX,::resampZ] )
    count = count + 1

test_a = test_a.permute((0,2,3,1))
test_u = test_u.permute((0,2,3,1))
toc()
################################################################

#print(train_a.shape)
#print(train_u.shape)
print(test_a.shape)
print(test_u.shape)

model = torch.load(modelname)
y_normalizer = torch.load(y_normalizer_name)
test_a = test_a.reshape(ntest,nx,nz,1,Nd0).repeat([1,1,1,Nd2,1])


# pad locations (x,y,t)
gridx = torch.tensor(np.linspace(0, 1, nx), dtype=torch.float)
gridx = gridx.reshape(1, nx, 1, 1, 1).repeat([1, 1, nz, Nd2, 1])
gridy = torch.tensor(np.linspace(0, 1, nz), dtype=torch.float)
gridy = gridy.reshape(1, 1, nz, 1, 1).repeat([1, nx, 1, Nd2, 1])
gridt = torch.tensor(np.linspace(0, 1, Nd2+1)[1:], dtype=torch.float)
gridt = gridt.reshape(1, 1, 1, Nd2, 1).repeat([1, nx, nz, 1, 1])
print('here at 1')


test_a = torch.cat((gridx.repeat([ntest,1,1,1,1]), gridy.repeat([ntest,1,1,1,1]),
                       gridt.repeat([ntest,1,1,1,1]), test_a), dim=-1)



test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)

t2 = default_timer()

print(test_a.shape)
print(test_u.shape)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

metrics=( np.zeros((epochs,3),dtype='float32') )

myloss = LpLoss(size_average=False)
#y_normalizer#.cuda()
ep=0;
tic()

for ep in range(epochs):
    #tic()


    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            #x, y = x.cuda(), y.cuda()
            it=0;
            out = model(x).view(batch_size, nx, nz, T)
            out = y_normalizer.decode(out)
            test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()


    t2 = default_timer()
    print('running inference ')
   
    uxS=[];
    pp=0 #ntest-batch_size
    uxS=test_u.data[pp+0,:,:,0].numpy()
    uxF=[];
    uxF=out.data[0,:,:,0].numpy()
    uzS=[];
    pp=0 #ntest-batch_size
    uzS=test_u.data[pp+0,:,:,1].numpy()
    uzF=[];
    uzF=out.data[0,:,:,1].numpy()
    
    Y,X = np.mgrid[0:600:10, 0:2000:10]
    U=uxS.T
    V=uzS.T
    cmap='rainbow'
    
    #mm=mm.reshape(Nx,Nz)
    vmax = np.max(uxS)
    vmin = np.min(uxS)
    
    fig = plt.figure(num=no,figsize=(20, 10)) ;
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
 
        
    
    mm1=[];
    mm1=uxF
    mm1[uxS ==0] = 'nan'
    qq=np.copy(mm1)
    plt.subplot(323)
    plt.pcolormesh(X,Y,mm1.T,vmin=-vmax*cc,vmax=cc*vmax,shading='gouraud',cmap=cmap);
    plt.colorbar(label='Velocity (m/s)');
    plt.streamplot(X,Y, U,V, density=0.5, color='k', linewidth=1, broken_streamlines=False)
    plt.title('Horizontal Wind Speed : FNO',fontsize=14);
    plt.xlabel('Distance (m)',fontsize=14);
    plt.ylabel('Height (m)',fontsize=14);
    
    mm1=[];
    mm1=uxS-uxF
    mm1[uxS ==0] = 'nan'
    plt.subplot(325)
    plt.pcolormesh(X,Y,mm1.T,vmin=-vmax*cc,vmax=cc*vmax,shading='gouraud',cmap=cmap);
    #plt.streamplot(X,Y, U,V, density=0.5, color='k', linewidth=1, broken_streamlines=False)    
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
    plt.title('Difference Vertical Wind Speed',fontsize=14);
    plt.xlabel('Distance (m)',fontsize=14);
    plt.ylabel('Height (m)',fontsize=14);

    plt.savefig(outputimage);
   
    
    sys.stdout.flush()


toc()

#########################################################################




