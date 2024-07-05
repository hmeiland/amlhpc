# turbine diameter 100 m
# turbine height 150 m
# wake effect stretch over 10-20 times the diameter
# pip install numpy-stl
# pip install fluidfoam

import numpy as np
from stl import mesh
from scipy.stats import powerlaw, powerlognorm, pareto, gamma
from scipy import signal
import matplotlib . pyplot as plt
  
def replace_text(string1,string2, filename):
    
    search_text = string1 
    replace_text = string2
 
    with open(filename, 'r') as file: 
      
        data = file.read() 
        data = data.replace(search_text, replace_text) 

    with open(filename, 'w') as file: 

        file.write(data) 

    print("Text replaced")
    
###############################################################
###############################################################
# generate synthetic dataset to test LSTM
dx=5
dy=5;
dz=10;
Lx=3500
Ly=250;
###############################################################
###############################################################

Nx=int(Lx/dx)
Ny=int(Ly/dy)
pad=50
Nz=1
pc = np.zeros((Nx*Ny,3), dtype=np.float32)
xlen = np.zeros((Nx,1), dtype=np.float32)
    
Nt=Nx+2*pad;
Nd=1;
ts = np.zeros((Nt,Nd), dtype=np.float32)
meants = np.zeros((1,Nd), dtype=np.float32)


f1=0.0001
f2=0.0200

a=1.5;
ts[:,0] = powerlaw.rvs(a, size=Nt)
meants[0,0]=np.mean(ts[:,0]);
b, a = signal.butter(3, [f2], btype='low')
tsf = signal.filtfilt(b, a, ts[:,0] ) 
ts[:,0] = tsf / np.linalg.norm(tsf)
ts[:,0]  = 50+400*(tsf-np.min(tsf))/(np.max(tsf)-np.min(tsf)) 

top=ts[pad:Nt-pad,:]


xx=int(700/dx)
h=int(top[xx,0]+120)
print('topo at 700 m is ',h-120)
print('turbine centre at 700 m is ',h)

filename="system/snappyHexMeshDict"
string1="snappy1"
string2=str(h-50)
replace_text(string1,string2, filename)
string1="snappy2"
string2=str(h+50)
replace_text(string1,string2, filename)
  

filename="system/topoSetDict"
string1="topo1"
string2=str(h)
replace_text(string1,string2, filename)

filename="constant/fvModels"
string1="fvmo1"
string2=str(h+35)
replace_text(string1,string2, filename)

#############################################################################
#############################################################################
k=0;
for i in range(0,Nx):
    xlen[i,0]=i*dx;
    for j in range(0,Ny):
        pc[k,0]=i*dx-50;
        pc[k,1]=j*dy-25;
        mult=1;
        if(j==0 or j==2):
            mult=0.8
        pc[k,2]=top[i,0] #*mult ;
        k=k+1;

aa = np.zeros( (len(xlen),2), dtype=np.float32 )
aa[:,0]=xlen[:,0]
aa[:,1]=top[:,0];
np.save('topo.npy', aa)    # .npy extension is added if not given

#############################################################################
# plot synthetic signals
'''
no=20
fig = plt.figure(num=no,figsize=(12, 8)) ;
plt.clf();
#plt.subplot(211)
plt.plot(xlen,top[:,0],  'r-', lw=5, alpha=0.6, label='ts0')
plt.title('Testing ')
plt.xlabel('Time')
plt.ylabel('Value')
plt.text(50,1,'Power Law Distribution',fontsize=12)
plt.grid('on')
plt.axis('equal')
#plt.xlim([0,1000])
'''

'''
fig = plt.figure(num=no+1,figsize=(8, 12)) ;
plt.clf();    
ax = plt.axes(projection='3d')
trisurf = ax.plot_trisurf(pc[:,0], pc[:,1], pc[:,2], 
                ax.view_init(60, 35), cmap='viridis', edgecolor='none');
''' 
import matplotlib.tri as mtri

x_all = pc[:,0]
y_all = pc[:,1]
z_all = pc[:,2]

tris = mtri.Triangulation(x_all, y_all)

data = np.zeros(len(tris.triangles), dtype=mesh.Mesh.dtype)
m = mesh.Mesh(data, remove_empty_areas=False)
m.x[:] = x_all[tris.triangles]
m.y[:] = y_all[tris.triangles]
m.z[:] = z_all[tris.triangles]

m.save('constant/geometry/pc_to_stl.stl')  
#############################################################################
'''
Nz=1000
dz=1
u = np.zeros((Nz), dtype=np.float32)
h = np.zeros((Nz), dtype=np.float32)

uref=8;
zref=10;
z0=0.1;
d=0;

kk=0.41;
    
for i in range(0,Nz):
    z=i*dz
    h[i]=z;
    zz = (z-d+z0)/z0
    us = uref*kk / np.log((zref+z0)/(z0))
    u[i] = np.log(zz) * (us/kk)
  

fig = plt.figure(num=no+2,figsize=(12, 8)) ;
plt.clf();
#plt.subplot(211)
plt.plot(u,h,  'r-', lw=5, alpha=0.6, label='ts0')
plt.title('Testing ')
plt.xlabel('U')
plt.ylabel('H')
plt.grid('on')
#plt.axis('equal')
#plt.xlim([0,1000])

'''
#############################################################################