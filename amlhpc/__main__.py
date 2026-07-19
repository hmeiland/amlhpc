import sys


def _use(vargs):
    import argparse

    from .context import ConnectionNotConfigured, use_profile

    parser = argparse.ArgumentParser(
        prog='amlhpc use', description='switch the current cluster profile')
    parser.add_argument('name', help='name of the cluster profile to make current')
    args = parser.parse_args(vargs)
    try:
        profile = use_profile(args.name)
    except ConnectionNotConfigured as error:
        print(error.message)
        return -1
    print("switched current cluster to '" + args.name + "' ("
          + profile['workspace'] + ')')
    return 0


def _clusters(vargs):
    import argparse

    from .context import load_config

    parser = argparse.ArgumentParser(
        prog='amlhpc clusters', description='list registered cluster profiles')
    parser.parse_args(vargs)
    config = load_config()
    clusters = config['clusters']
    if not clusters:
        print("no cluster profiles yet; add one with 'deploy connect' or 'deploy init'")
        return 0
    current = config['current']
    print('CURRENT  NAME            WORKSPACE')
    for name in sorted(clusters):
        marker = '*' if name == current else ' '
        print('   ' + marker + '     ' + name.ljust(15) + ' '
              + clusters[name]['workspace'])
    return 0


def _commands():
    from .slurm.sbatch import sbatch
    from .slurm.srun import srun
    from .slurm.sinfo import sinfo
    from .slurm.squeue import squeue
    from .slurm.scancel import scancel
    from .slurm.sacct import sacct
    from .slurm.sstat import sstat
    from .slurm.sattach import sattach
    from .pbs.qsub import qsub
    from .pbs.qstat import qstat, qdel
    from .lsf.bjobs import bjobs, bkill
    from .lsf.bsub import bsub
    from .container import container
    from .deploy import deploy
    from .dask.scheduler import dask_scheduler_up
    from .dask.worker import dask_up
    from .dask.down import dask_down

    return {
        'sbatch': sbatch,
        'srun': srun,
        'sinfo': sinfo,
        'squeue': squeue,
        'scancel': scancel,
        'sacct': sacct,
        'sstat': sstat,
        'sattach': sattach,
        'qsub': qsub,
        'qstat': qstat,
        'qdel': qdel,
        'bjobs': bjobs,
        'bkill': bkill,
        'bsub': bsub,
        'container': container,
        'deploy': deploy,
        'dask-scheduler-up': dask_scheduler_up,
        'dask-up': dask_up,
        'dask-down': dask_down,
        'use': _use,
        'clusters': _clusters,
    }


def main(vargs=None):
    argv = list(sys.argv[1:] if vargs is None else vargs)

    commands = _commands()

    if not argv or argv[0] in ('-h', '--help'):
        print('usage: amlhpc <command> [<args>]')
        print()
        print('amlhpc: a -just enough- Slurm/PBS experience on Azure Machine Learning')
        print()
        print('commands:')
        for name in commands:
            print('  ' + name)
        print()
        print("run 'amlhpc <command> --help' for command-specific options")
        return 0 if argv else -1

    command = argv[0]
    if command not in commands:
        print("amlhpc: unknown command '" + command + "'")
        print("run 'amlhpc --help' for the list of commands")
        return -1

    return commands[command](argv[1:])


if __name__ == '__main__':
    sys.exit(main())
