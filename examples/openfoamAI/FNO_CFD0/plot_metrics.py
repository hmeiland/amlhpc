'''

./run_jobs1.sh 0 1000 > log1 &
./run_jobs2.sh 1001 2000 > log2 &
./run_jobs3.sh 2001 3000 > log3 &
./run_jobs4.sh 3001 4000 > log4 &
./run_jobs5.sh 4001 5000 > log5 &
./run_jobs6.sh 5001 6000 > log6 &
./run_jobs7.sh 6001 7000 > log7 &
./run_jobs8.sh 7001 8000 > log8 &
./run_jobs9.sh 8001 9000 > log9 &
./run_jobs10.sh 9001 10000 > log10 &

./run_jobs1.sh 10000 11000 > log1 &
./run_jobs2.sh 11001 12000 > log2 &
./run_jobs3.sh 12001 13000 > log3 &
./run_jobs4.sh 13001 14000 > log4 &
./run_jobs5.sh 14001 15000 > log5 &
./run_jobs6.sh 15001 16000 > log6 &
./run_jobs7.sh 16001 17000 > log7 &
./run_jobs8.sh 17001 18000 > log8 &
./run_jobs9.sh 18001 19000 > log9 &
./run_jobs10.sh 19001 20000 > log10 &


'''

import numpy as np
import sys,time
import matplotlib.pyplot as plt
from timeit import default_timer
import scipy.io
import pandas as pd

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
#metrics0=( np.zeros((50,3),type='float32') )
metrics1=np.load('metrics_t7.npy')
metrics2=np.load('metrics_t8.npy')

fig = plt.figure(num=109,figsize=(6, 4)) ;
plt.clf();
plt.subplot(211)
plt.plot(metrics1[:,1]/metrics1[0,1].max(),'r')  #,vmin=-vmax*cc,vmax=cc*vmax);
plt.plot(metrics1[:,2]/metrics1[0,2].max(),'g--')#,vmin=-vmax*cc,vmax=cc*vmax);
#plt.semilogy(metrics1[:,1]/metrics1[0,1].max(),'r')  #,vmin=-vmax*cc,vmax=cc*vmax);
#plt.semilogy(metrics1[:,2]/metrics1[0,2].max(),'g--')#,vmin=-vmax*cc,vmax=cc*vmax);
plt.axis([0 ,200,0.2,1])
plt.grid()

plt.subplot(212)    
plt.plot(metrics2[:,1]/metrics2[0,1].max(),'r')  #,vmin=-vmax*cc,vmax=cc*vmax);
plt.plot(metrics2[:,2]/metrics2[0,2].max(),'g--')#,vmin=-vmax*cc,vmax=cc*vmax);
#plt.semilogy(metrics2[:,1]/metrics2[0,1].max(),'r')  #,vmin=-vmax*cc,vmax=cc*vmax);
#plt.semilogy(metrics2[:,2]/metrics2[0,2].max(),'g--')#,vmin=-vmax*cc,vmax=cc*vmax);
plt.axis([0 ,200,0.2,1])
plt.grid()
