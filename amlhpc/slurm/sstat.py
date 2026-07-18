def sstat(vargs=None):
    import sys
    from ..jobcontrol import show_job_stats
    show_job_stats("sstat", sys.argv[1:] if vargs is None else vargs)
