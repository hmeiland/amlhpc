def sattach(vargs=None):
    import sys
    from ..jobcontrol import attach_job
    attach_job("sattach", sys.argv[1:] if vargs is None else vargs)
