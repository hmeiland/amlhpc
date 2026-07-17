def scancel(vargs=None):
    import sys
    from ..jobcontrol import cancel_job
    cancel_job("scancel", sys.argv[1:] if vargs is None else vargs)
