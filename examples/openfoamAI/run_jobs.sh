#!/bin/bash

echo "Run jobs"

ls

for i in {0..1000} 
do
   date
   echo ".....running simulation$i "
   rm Simu$i -rf
   cp ControlSimu Simu$i -r
   cd Simu$i
   #pwd 
   #ls
   python set_up_job.py
   ./Allrun
   python extract_model_result.py
   cp model.npy ../Results/model$i.npy
   cp ux.npy ../Results/ux$i.npy   
   cp uz.npy ../Results/uz$i.npy
   cp topo.npy  ../Results/topo$i.npy 
   cd ../
   date
   echo ".....finished simulation$i "
   
done
