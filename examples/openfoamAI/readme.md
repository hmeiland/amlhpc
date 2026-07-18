# OpenFOAM + AI (Fourier Neural Operator surrogate)

An end-to-end example of training a machine-learning surrogate for a CFD problem:
generate a dataset of OpenFOAM wind-turbine wake simulations, then train a Fourier
Neural Operator (FNO) that predicts the flow field from the terrain/turbine model.

This example is not a single `sbatch` job; it is a two-stage workflow whose scripts
still carry the author's absolute paths and OpenFOAM environment assumptions (see the
caveats below). Treat it as a worked reference rather than a turn-key run.

## Layout

```
run_jobs.sh                 # stage 1 driver: loops many OpenFOAM simulations into a dataset
ControlSimu/                # the OpenFOAM case template that each run is copied from
  Allrun                    # blockMesh -> decomposePar -> snappyHexMesh -> topoSet -> solve -> reconstruct
  Allclean                  # clean the case
  set_up_job.py             # randomises the case geometry (turbine/terrain) per run
  extract_model_result.py   # reads U/p at time 1000, interpolates to a grid, writes model/ux/uz .npy
  0/  constant/  system/    # standard OpenFOAM case directories
FNO_CFD0/                   # stage 2: the FNO surrogate
  FNO_fn1.py                # FNO layers / model definition
  fno_cfd0.py               # training entry point (reads ../Results, trains the operator)
  apply_model1.py           # inference on a trained model
  apply_model2.py           # inference variant
  utilities3.py             # loss/normalisation helpers
  plot_cfd0.py              # plot fields
  plot_metrics.py           # plot training metrics
  inference_snap.jpg        # example inference output
```

## Stage 1 — build the dataset with OpenFOAM

`run_jobs.sh` copies the `ControlSimu` case into `Simu<i>` for each index, randomises the
geometry, runs the OpenFOAM `Allrun`, extracts the result to NumPy arrays, and collects them
under `Results/`:

```bash
for i in {0..1000}
do
   cp ControlSimu Simu$i -r
   cd Simu$i
   python set_up_job.py            # randomise turbine/terrain geometry for this sample
   ./Allrun                        # run the OpenFOAM case (parallel)
   python extract_model_result.py  # -> model.npy, ux.npy, uz.npy (topo.npy from set_up_job.py)
   cp model.npy ../Results/model$i.npy
   cp ux.npy    ../Results/ux$i.npy
   cp uz.npy    ../Results/uz$i.npy
   cp topo.npy  ../Results/topo$i.npy
   cd ../
done
```

OpenFOAM itself comes from the EESSI software stack, exactly like the [OpenFOAM](../OpenFOAM/readme.md),
[GROMACS](../GROMACS/README.md) and [QuantumESPRESSO](../QuantumESPRESSO/README.md) examples. Prepend the
EESSI bootstrap to your runscript before invoking `run_jobs.sh`:

```bash
sudo mount -t cvmfs software.eessi.io /cvmfs/software.eessi.io
source /cvmfs/software.eessi.io/versions/2023.06/init/bash
ml load OpenFOAM
source $FOAM_BASH
```

The Python helpers additionally need `numpy`, `scipy`, `matplotlib`, `numpy-stl` and
`fluidfoam` (`pip install numpy-stl fluidfoam`).

Submit the dataset generation with the `simple` datamover so the whole working directory
(the `ControlSimu` template and the driver) is uploaded with the job:

```bash
sbatch -p hbv2 --datamover=simple ./runscript.sh
```

(where `runscript.sh` is the EESSI bootstrap above followed by `./run_jobs.sh`).

## Stage 2 — train the FNO surrogate

Once `Results/` is populated, `FNO_CFD0/fno_cfd0.py` trains the Fourier Neural Operator on the
`(model, topo) -> (ux, uz)` pairs, and `apply_model1.py` / `apply_model2.py` run inference. These
scripts are PyTorch-based; run them in an environment/container that provides `torch` (e.g. a GPU
partition), for example:

```bash
sbatch -p <gpu-partition> --datamover=simple ./train_runscript.sh
```

## Caveats

- The FNO scripts contain a hardcoded author path
  (`sys.path.insert(0, '/mnt/c/Users/.../FNO_CFD0')`) and both stages assume specific grid sizes,
  the OpenFOAM result time `1000`, and a `../Results` layout. Adjust these to your own paths and
  case before running.
- `run_jobs.sh` loops 0..1000 and `plot_metrics.py` references sharded `run_jobs*.sh` drivers for
  much larger sweeps; scale the range to your capacity.
- This is a reference workflow illustrating how to couple OpenFOAM data generation on AmlCompute
  with ML training via amlhpc; it is not a self-contained regression test.
