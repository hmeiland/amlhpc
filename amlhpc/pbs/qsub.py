def qsub(vargs=None):
    import sys
    import os
    import argparse
    import re

    argv = sys.argv[1:] if vargs is None else list(vargs)

    parser = argparse.ArgumentParser(
        description='qsub: submit jobs to Azure Machine Learning (PBS-style front-end for sbatch)',
        epilog='submit a script with: qsub -q f16s runscript.sh. '
               'submit a command with: qsub -q f16s hostname')
    parser.prog = "qsub"
    parser.add_argument('-q', '--queue', default=None, type=str,
                        help='PBS queue/destination to run in; maps to the sbatch partition. Use <sinfo> to view available queues')
    parser.add_argument('-l', '--resource', action='append', default=[], type=str,
                        help='resource request, e.g. -l nodes=4 or -l select=4 or -l nodes=2:ppn=8 (node count maps to sbatch --nodes)')
    parser.add_argument('-N', '--name', default=None, type=str,
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
        print("Missing: provide a script to submit (qsub -q f16s runscript.sh) or a command (qsub -q f16s hostname)")
        exit(-1)

    # PBS resource string: node count from nodes=<N> or select=<N>.
    nodes = None
    for resource in args.resource:
        match = re.search(r'(?:nodes|select)=(\d+)', resource)
        if match:
            nodes = int(match.group(1))
            if args.verbose:
                print("qsub: node count from resource request: " + str(nodes))

    sbatch_argv = ['-p', args.queue]
    if nodes is not None:
        sbatch_argv += ['-N', str(nodes)]
    if args.container is not None:
        sbatch_argv += ['--container', args.container]
    if args.environment is not None:
        sbatch_argv += ['-e', args.environment]
    if args.verbose:
        sbatch_argv += ['-v'] * args.verbose

    # PBS submits a script when a single existing file is given; otherwise the
    # trailing tokens form the command line (qsub run.sh / qsub hostname).
    if len(command) == 1 and os.path.isfile(command[0]):
        sbatch_argv.append(command[0])
    else:
        sbatch_argv += ['--wrap', ' '.join(command)]

    if args.name is not None and args.verbose:
        print("qsub: job name '" + args.name + "' is informational; AML assigns the JOBID")

    from ..slurm.sbatch import sbatch
    return sbatch(sbatch_argv)
