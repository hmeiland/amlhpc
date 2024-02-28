
# pip install fluidfoam

import numpy as np
import matplotlib . pyplot as plt
from scipy.interpolate import griddata

from fluidfoam import readvector
from fluidfoam import readmesh
from fluidfoam import readscalar

sol = './'
x, y, z = readmesh(sol)

timename = '1000'
vel1 = readvector(sol, timename, 'U')
press1 = readscalar(sol, timename, 'p')


# Number of division for linear interpolation
ngridx = 200
ngridy = 20
ngridz = 60

# Interpolation grid dimensions
xinterpmin = 0
xinterpmax = 2000
yinterpmin = 0
yinterpmax = 200
zinterpmin = 0
zinterpmax = 1000

# Interpolation grid
xi = np.linspace(xinterpmin, xinterpmax, ngridx)
yi = np.linspace(yinterpmin, yinterpmax, ngridy)
zi = np.linspace(zinterpmin, zinterpmax, ngridz)

lenX=x.size;

vel2=vel1[:,0:lenX]
# Structured grid creation
xinterp, yinterp , zinterp = np.meshgrid(xi, yi, zi)


# Interpolation of scalra fields and vector field components
#p_i = griddata((x, y,z), press1, (xinterp, yinterp,zinterp), method='linear')
vx_i = griddata((x, y,z), vel2[0, :], (xinterp, yinterp,zinterp), method='nearest')
#vy_i = griddata((x, y,z), vel2[1, :], (xinterp, yinterp,zinterp), method='linear')
vz_i = griddata((x, y,z), vel2[2, :], (xinterp, yinterp,zinterp), method='nearest')

topo = np.load('topo.npy')
t_i = griddata((topo[:,0]), topo[:,1], (xi), method='nearest')

ux=np.copy( vx_i[10,:,:] )
uz=np.copy( vz_i[10,:,:] )
mo=np.copy( vz_i[10,:,:] )
mo[:,:]=0
dz=1000/ngridz
for i in range(0,ngridx):
    ux[i,0:int(t_i[i]/dz)]=0
    uz[i,0:int(t_i[i]/dz)]=0
    mo[i,0:int(t_i[i]/dz)]=1
    
i=70
mo[i,int(t_i[i]/dz)+7]=1
mo[i,int(t_i[i]/dz)+8]=1
mo[i,int(t_i[i]/dz)+9]=1
mo[i,int(t_i[i]/dz)+10]=1

np.save('model.npy', mo) #
np.save('ux.npy', ux)    #
np.save('uz.npy', uz)    #

###############################################################
###############################################################

no=30
fig = plt.figure(num=no,figsize=(12, 8)) ;
plt.clf();
plt.subplot(311)
plt.pcolormesh(np.transpose(mo), vmin=0,vmax=1,shading='gouraud'); #, cmap="RdGy")
plt.plot(t_i/16.667); #, cmap="RdGy")
plt.colorbar();

plt.subplot(312)
plt.pcolormesh(np.transpose(ux), vmin=4,vmax=25,shading='gouraud'); #, cmap="RdGy")
plt.colorbar();

plt.subplot(313)
plt.pcolormesh(np.transpose(uz), vmin=-2,vmax=2,shading='gouraud'); #, cmap="RdGy")
plt.colorbar();

plt.savefig('results.png')

#############################################################################
#############################################################################