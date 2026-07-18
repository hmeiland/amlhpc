def sacct(vargs=None):
    import sys
    from ..jobcontrol import show_job_status
    show_job_status("sacct", sys.argv[1:] if vargs is None else vargs)
