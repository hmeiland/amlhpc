sudo mount -t cvmfs software.eessi.io /cvmfs/software.eessi.io
source /cvmfs/software.eessi.io/versions/2023.06/init/bash

ml load TensorFlow
cd outputs

#python -c "import tensorflow as tf; tf.keras.datasets.cifar10.load_data()"
python ../cifar.py

