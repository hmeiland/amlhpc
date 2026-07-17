def bsub(vargs=None):
    import sys
    import argparse

    argv = sys.argv[1:] if vargs is None else list(vargs)

    parser = argparse.ArgumentParser(
        description='bsub: submit jobs to Azure Machine Learning (LSF-style front-end for sbatch)',
        epilog='the job command follows the options: bsub -q f16s hostname. '
               'a script is submitted with: bsub -q f16s script.sh')
    parser.prog = "bsub"
    parser.add_argument('-q', '--queue', default=None, type=str,
                        help='LSF queue to run in; maps to the sbatch partition. Use <sinfo> to view available queues')
    parser.add_argument('-n', '--num-slots', dest='num_slots', default=None, type=int,
                        help='number of nodes to use for the job (maps to sbatch --nodes)')
    parser.add_argument('-J', '--job-name', dest='job_name', default=None, type=str,
                        help='job name (informational; AML assigns the JOBID)')
    parser.add_argument('--container', default=None, type=str,
                        help='container environment for the job to run in')
    parser.add_argument('-e', '--environment', default=None, type=str,
                        help='Azure Machine Learning environment, may use @latest')
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help='provide output on found settings and job properties')
    args, command = parser.parse_known_args(argv)

    if args.queue is None:
        print("Missing: provide the queue to run the job with -q (see <sinfo> for available queues)")
        exit(-1)

    if not command:
        print("Missing: provide the command (or script) to execute after the bsub options, e.g. bsub -q f16s hostname")
        exit(-1)

    sbatch_argv = ['-p', args.queue]
    if args.num_slots is not None:
        sbatch_argv += ['-N', str(args.num_slots)]
    if args.container is not None:
        sbatch_argv += ['--container', args.container]
    if args.environment is not None:
        sbatch_argv += ['-e', args.environment]
    if args.verbose:
        sbatch_argv += ['-v'] * args.verbose

    # LSF submits a script when a single existing file is given; otherwise the
    # trailing tokens form the command line (bsub hostname / bsub echo hi).
    import os
    if len(command) == 1 and os.path.isfile(command[0]):
        sbatch_argv.append(command[0])
    else:
        sbatch_argv += ['--wrap', ' '.join(command)]

    if args.job_name is not None and args.verbose:
        print("bsub: job name '" + args.job_name + "' is informational; AML assigns the JOBID")

    from ..slurm.sbatch import sbatch
    return sbatch(sbatch_argv)
