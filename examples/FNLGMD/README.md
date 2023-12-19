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
num_epochs: 20 

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

initial_pop_size: 100
max_population: 500

#FNL JTNN specific
vocab_path: './all_vocab.txt'
model_path: './model.epoch-35'
```

Since all the input files will be copied to the cluster vm, they should not reside in different directories: 
so make sure to copy the zinc_smiles.txt file from the other example directory. The output directory is set to
the default AML output directory: ./outputs/.

# running with a cutom environment

   
