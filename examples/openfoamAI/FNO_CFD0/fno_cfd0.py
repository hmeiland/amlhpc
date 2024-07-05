

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
# configs
################################################################

no=10;
filename0 = '../Results';   # model
filename1 = '../Results';  # result
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

ntrain = 900
ntest = 100
skip= 0 ;
weight_ux=0.15

modes = 2
width = 24

batch_size = 20 #  used ~ 20
batch_size2 = batch_size

epochs = 100
learning_rate = 0.005 # 0.001
scheduler_step = 40
scheduler_gamma = 0.5

#print(epochs, learning_rate, scheduler_step, scheduler_gamma)
# save
err_out = 'metrics_t2'
path = 'model_t2'
# path = 'ns_fourier_V100_N'+str(ntrain)+'_ep' + str(epochs) + '_m' + str(modes) + '_w' + str(width)
path_model = 'model/'+path
path_train_err = 'results/'+path+'train.txt'
path_test_err = 'results/'+path+'test.txt'
path_image = 'image/'+path

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
train_a=torch.tensor( np.zeros((ntrain, Nd0,nx,nz),dtype='float32') ) 
train_u=torch.tensor( np.zeros((ntrain, Nd2,Nx,Nz),dtype='float32') )


test_a=torch.tensor( np.zeros((ntest, Nd0,nx,nz),dtype='float32') ) 
test_u=torch.tensor( np.zeros((ntest, Nd2,Nx,Nz),dtype='float32') )
count = 0

print(train_a.shape)
print(train_u.shape)
print(test_a.shape)
print(test_u.shape)

Umax=( np.zeros((ntrain+ntest,3),dtype='float32') )

###############################################################################
tic()
print('Loading data ...')

for lt in range(0,ntrain+ntest):
   
    if(lt%200==0):
        print(lt)
        
    ff = filename0+'/model'+str(lt+skip+1)+'.npy'
    check_file = os.path.isfile(ff)
    if(check_file==True):
        model = np.load(ff)
        
        #if(lt%10==0):
        #    print(ff)
        shapeM=(1,NxM,NzM)
        M[0,:,:] = model
        
        ff = filename1+'/ux'+str(lt+skip+1)+'.npy'
        ux = np.load(ff)
        ff = filename1+'/uz'+str(lt+skip+1)+'.npy'
        uz = np.load(ff)
        
        shape=(Nd0,NxU,NzU)
        U[0,:,:] = ux[:,:]*weight_ux
        U[1,:,:] = uz[:,:]
    else:
        print('    No file',ff)
        
    Umax[lt][0] = np.abs(U[0,:,:]).max();
    if(np.absolute(Umax[lt][0])>80):
        print('0 ',lt+skip,' ')
        
    Umax[lt][1] = np.abs(U[1,:,:]).max();
    if(Umax[lt][1]>50):
        print('1 ',lt+skip,' ')  
   
    Umax[lt][2] = np.abs(M[0,:,:]).max();
    if(Umax[lt][2]>2):
        print('3 ',lt+skip,' ') 

    ###############################################################################
    # pcolor plot
    '''
    mm=[];
    mm=M[0,:,:]
    #mm=mm.reshape(Nx,Nz)
    vmax = np.max(mm)
    vmin = np.min(mm)
    fig = plt.figure(num=no,figsize=(8, 5)) ;
    plt.clf();
    plt.pcolormesh(mm.T)#,vmin=-vmax*cc,vmax=cc*vmax);
    plt.colorbar();
    plt.pause(0.2)

    mm=[];
    mm=U[1,:,:]
    #mm=mm.reshape(Nx,Nz)
    vmax = np.max(mm)
    vmin = np.min(mm)
    fig = plt.figure(num=no+1,figsize=(8, 5)) ;
    plt.clf();
    plt.pcolormesh(mm.T)#,vmin=-vmax*cc,vmax=cc*vmax);
    plt.colorbar();
    plt.pause(0.2)
    '''
    
    if lt<ntrain:
        train_a[lt,:,:nx,:]=torch.tensor( M[:,::resampXm,::resampZm] )
        train_u[lt,0,:,:]=torch.tensor( U[0,::resampX,::resampZ] )
        train_u[lt,1,:,:]=torch.tensor( U[1,::resampX,::resampZ] )
    elif lt>=ntrain:
        test_a[count,:,:nx,:]=torch.tensor( M[:,::resampXm,::resampZm] )
        test_u[count,0,:,:]=torch.tensor( U[0,::resampX,::resampZ] )
        test_u[count,1,:,:]=torch.tensor( U[1,::resampX,::resampZ] )
        count = count + 1

fig = plt.figure(num=no+2,figsize=(8, 5)) ;
plt.clf();
plt.subplot(311)
plt.plot(Umax[:,0],'r')  #,vmin=-vmax*cc,vmax=cc*vmax);

plt.subplot(312)
plt.plot(Umax[:,1],'g')  #,vmin=-vmax*cc,vmax=cc*vmax);
plt.pause(0.05)

plt.subplot(313)
plt.plot(Umax[:,2],'b')  #,vmin=-vmax*cc,vmax=cc*vmax);
plt.pause(0.05)

#plt.savefig('test1_data.jpg');

train_a = train_a.permute((0,2,3,1))
train_u = train_u.permute((0,2,3,1))
test_a = test_a.permute((0,2,3,1))
test_u = test_u.permute((0,2,3,1))
toc()
################################################################

print(train_a.shape)
print(train_u.shape)
print(test_a.shape)
print(test_u.shape)

#assert (nx == train_u.shape[1])
#assert (T == train_u.shape[3])


#a_normalizer = UnitGaussianNormalizer(train_a)
#train_a = a_normalizer.encode(train_a)
#test_a = a_normalizer.encode(test_a)

y_normalizer = UnitGaussianNormalizer(train_u)
train_u = y_normalizer.encode(train_u)
#test_u = y_normalizer.encode(test_u)

#torch.save(a_normalizer,'anormal2');
torch.save(y_normalizer,'ynormal2');

train_a = train_a.reshape(ntrain,nx,nz,1,Nd0).repeat([1,1,1,Nd2,1])
test_a = test_a.reshape(ntest,nx,nz,1,Nd0).repeat([1,1,1,Nd2,1])


# pad locations (x,y,t)
gridx = torch.tensor(np.linspace(0, 1, nx), dtype=torch.float)
gridx = gridx.reshape(1, nx, 1, 1, 1).repeat([1, 1, nz, Nd2, 1])
gridy = torch.tensor(np.linspace(0, 1, nz), dtype=torch.float)
gridy = gridy.reshape(1, 1, nz, 1, 1).repeat([1, nx, 1, Nd2, 1])
gridt = torch.tensor(np.linspace(0, 1, Nd2+1)[1:], dtype=torch.float)
gridt = gridt.reshape(1, 1, 1, Nd2, 1).repeat([1, nx, nz, 1, 1])
print('here at 1')

train_a = torch.cat((gridx.repeat([ntrain,1,1,1,1]), gridy.repeat([ntrain,1,1,1,1]),
                       gridt.repeat([ntrain,1,1,1,1]), train_a), dim=-1)

test_a = torch.cat((gridx.repeat([ntest,1,1,1,1]), gridy.repeat([ntest,1,1,1,1]),
                       gridt.repeat([ntest,1,1,1,1]), test_a), dim=-1)

'''
mm=[];
mm=train_a.data[3,:,:,10,3].numpy()
#mm=mm.reshape(Nx,Nz)
vmax = np.max(mm)
vmin = np.min(mm)
fig = plt.figure(num=no+3,figsize=(8, 5)) ;
plt.clf();
plt.pcolormesh(mm.T)#,vmin=-vmax*cc,vmax=cc*vmax);
plt.colorbar();
'''

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)

t2 = default_timer()

#print('preprocessing finished, time used:', t2-t1)
#device = torch.device('cuda')

################################################################
# training and evaluation
################################################################

print(train_a.shape)
print(train_u.shape)
print(test_a.shape)
print(test_u.shape)

model = FNO3d(modes, modes, modes, width)

#print(count_params(model))
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

metrics=( np.zeros((epochs,3),dtype='float32') )

myloss = LpLoss(size_average=False)
#y_normalizer#.cuda()
ep=0;
tic()

for ep in range(epochs):
    #tic()
    model.train()
    t1 = default_timer()
    train_mse = 0
    train_l2 = 0
    it=0;
    for x, y in train_loader:
        #x, y = x.cuda(), y.cuda()
        if(it%2==0):
            print(it, end=" ")
        it = it +1
        optimizer.zero_grad()
        out = model(x).view(batch_size, nx, nz, T)

        #mse = F.mse_loss(out, y, reduction='mean')
        # mse.backward()

        y = y_normalizer.decode(y)
        out = y_normalizer.decode(out)
        l2 = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
        l2.backward()

        optimizer.step()
        #train_mse += mse.item()
        train_l2 += l2.item()

    scheduler.step()

    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            #x, y = x.cuda(), y.cuda()
            it=0;
            out = model(x).view(batch_size, nx, nz, T)
            out = y_normalizer.decode(out)
            test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()
            
    train_mse /= len(train_loader)
    train_l2 /= ntrain
    test_l2 /= ntest

    t2 = default_timer()
    #print(ep, t2-t1, train_mse, train_l2, test_l2)
    print(' ')
    print('epoch=',ep,' time=' ,t2-t1,'L2=',train_l2,'L2 test=',test_l2)

    metrics[ep][0] = t2-t1
    metrics[ep][1] = train_l2
    metrics[ep][2] = test_l2

    
    #toc()

    mm=[];
    pp=ntest-batch_size
    mm=test_u.data[pp+0,:,:,0].numpy()
    #mm=mm.reshape(Nx,Nz)
    vmax = np.max(mm)
    vmin = np.min(mm)
    
    fig = plt.figure(num=no+3,figsize=(8, 5)) ;
    plt.clf();
    plt.subplot(321)
    plt.pcolormesh(mm.T)#,vmin=-vmax*cc,vmax=cc*vmax);
    plt.colorbar();
    #plt.title(ep);
    plt.pause(0.05)
    np.save('Results_uxSimu_t2',mm)
    
    mm1=[];
    mm1=out.data[0,:,:,0].numpy()
    #mm=mm.reshape(Nx,Nz)
    vmax = np.max(mm1)
    vmin = np.min(mm1)
    plt.subplot(323)
    plt.pcolormesh(mm1.T)#,vmin=-vmax*cc,vmax=cc*vmax);
    plt.colorbar();
    plt.pause(0.05)
    np.save('Results_uxFNO_t2',mm1)
    
    plt.subplot(325)
    plt.pcolormesh(mm.T-mm1.T)#,vmin=-vmax*cc,vmax=cc*vmax);
    plt.colorbar();
    plt.pause(0.05)
 
    mm=[];
    pp=ntest-batch_size
    mm=test_u.data[pp+0,:,:,1].numpy()
    vmax = np.max(mm)
    vmin = np.min(mm)
    plt.subplot(322)
    plt.pcolormesh(mm.T)#,vmin=-vmax*cc,vmax=cc*vmax);
    plt.colorbar();
    plt.pause(0.05)
    np.save('Results_uzSimu_t2',mm)
    
    mm1=[];
    mm1=out.data[0,:,:,1].numpy()
    #mm=mm.reshape(Nx,Nz)
    vmax = np.max(mm1)
    vmin = np.min(mm1)
    plt.subplot(324)
    plt.pcolormesh(mm1.T)#,vmin=-vmax*cc,vmax=cc*vmax);
    plt.colorbar();
    plt.pause(0.05)
    np.save('Results_uzFNO_t2',mm1)
    
    plt.subplot(326)
    plt.pcolormesh(mm.T-mm1.T)#,vmin=-vmax*cc,vmax=cc*vmax);
    plt.colorbar();
    plt.pause(0.05) 
    plt.savefig('test2_resultsA.jpg');
      
    fig = plt.figure(num=no+4,figsize=(8, 5)) ;
    plt.clf();
    plt.plot(metrics[:,1]/metrics[0,1].max(),'r')  #,vmin=-vmax*cc,vmax=cc*vmax);
    plt.plot(metrics[:,2]/metrics[0,2].max(),'g--')#,vmin=-vmax*cc,vmax=cc*vmax);
    plt.pause(0.05)
    #plt.savefig('test2_resultsB.jpg');
    
    print(ep,' saving model..'); 
    sys.stdout.flush()
    torch.save(model, path_model)
    np.save(err_out,metrics);
    
torch.save(model, path_model)
toc()

#########################################################################




