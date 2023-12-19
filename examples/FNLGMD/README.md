# FNLGMD

This software is intended to act as a framework for generative molecular design research and can be found here: https://github.com/SeanTBlack/FNLGMD

# preparing config and input files

To run the examples, checkout or clone the github repository in the working directory.
Go to the LogP_JTVAE example directory and tweak the config file to point to the relative paths for the input files:

```
model_type: 'jtnn-fnl'

smiles_input_file: './zinc_smiles.txt'
output_directory: './outputs/'
scorer_type: 'LogPTestCase'
num_epochs: 2 

#Selection/Optimization params
optimizer_type: 'geneticoptimizer'
tourn_size: 15
mate_prob: 0.7
mutate_prob: 0.1
mutation_std: 1
max_clones: 1
optima_type: "maxima"
selection_type: "tournament"
elite_perc: 0

initial_pop_size: 10
max_population: 10

#FNL JTNN specific
vocab_path: './all_vocab.txt'
model_path: './model.epoch-35'
```

Since all the input files will be copied to the cluster vm, they should not reside in different directories: 
so make sure to copy the zinc_smiles.txt file from the other example directory. The output directory is set to
the default AML output directory: ./outputs/.

The next step is to create the runscript.sh. This is tweaked a little but from the run_gmd.sh which can be found in the root of the repository.
```
export PATH=/opt/conda/envs/env/bin:$PATH
source activate gmd
python /FNLGMD/source/main.py -config config.yaml
```

The content of the working directory should now look like this:
```
(azureml_py38) azureuser@humeilan1:~/cloudfiles/code/Users/humeilan/FNLGMD/examples/LogP_JTVAE$ ls -alh
total 44M
drwxrwxrwx 2 root root    0 Dec 18 19:19 .
drwxrwxrwx 2 root root    0 Dec 18 19:19 ..
-rwxrwxrwx 1 root root  251 Dec 18 19:19 README.md
-rwxrwxrwx 1 root root  525 Dec 18 19:19 Slurm-Run-Loop
-rwxrwxrwx 1 root root  11M Dec 18 19:19 all.txt
-rwxrwxrwx 1 root root 9.5K Dec 18 19:19 all_vocab.txt
-rwxrwxrwx 1 root root  471 Dec 19 07:14 config.yaml
-rwxrwxrwx 1 root root 2.0K Dec 18 19:19 example.out
-rwxrwxrwx 1 root root  23M Dec 18 19:19 model.epoch-35
-rwxrwxrwx 1 root root  112 Dec 19 07:14 runscript.sh
-rwxrwxrwx 1 root root  11M Dec 18 20:30 zinc_smiles.txt
```

# running with a custom environment

The above example input file requires the latest version of the code to run, hence the need to build the docker container ourselves.
This can easily be done based on the Dockerfile and both dependency files provided in the repo. Create a new Environment, set the name to fnlgmd
and select "Create a new docker context". Add the 3 required files and finish. The build will take a few minutes.


The job can now be submitted with:
```
$ sbatch -p f16s --environment="fnlgmd@latest" --datamover=simple ./runscript.sh
Uploading LogP_JTVAE (45.82 MBs): 100%|████████████████████████████████████████████| 45816548/45816548 [00:00<00:00, 118097961.49it/s]
tough_candle_ld3n0b2lxp
```

This should be a very quick job and the result can be found in the jobs "Outputs and logs" tab: 
