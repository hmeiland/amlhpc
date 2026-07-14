from argparse import ArgumentParser
from logging import getLogger
from . import squeue, sbatch, sinfo

log = getLogger('amlhpc')

parser = ArgumentParser('amlhpc', description='amlhpc: slurm APIs for Azure Machine Learning', add_help=False)
parser.add_argument('command', help='sub-command to run', choices=['squeue', 'sbatch', 'sinfo'])

args, extra_args = parser.parse_known_args()
if args.command == 'squeue':
    log.info("squeue")
    squeue(extra_args)
elif args.command == 'sbatch':
    log.info("sbatch")
    sbatch(extra_args)
elif args.command == 'sinfo':
    log.info("sinfo")
    sinfo(extra_args)
else:
    log.info("Unknown command")
    exit(-1)
