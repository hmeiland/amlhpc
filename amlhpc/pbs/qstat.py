def qstat(vargs=None):
    from ..jobcontrol import list_jobs
    list_jobs("qstat")


def qdel(vargs=None):
    import sys
    from ..jobcontrol import cancel_job
    cancel_job("qdel", sys.argv[1:] if vargs is None else vargs)
