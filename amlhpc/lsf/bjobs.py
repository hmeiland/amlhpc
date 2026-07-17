def bjobs(vargs=None):
    from ..jobcontrol import list_jobs
    list_jobs("bjobs")


def bkill(vargs=None):
    import sys
    from ..jobcontrol import cancel_job
    cancel_job("bkill", sys.argv[1:] if vargs is None else vargs)
