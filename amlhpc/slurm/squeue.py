def squeue(vargs=None):
    from ..jobcontrol import get_ml_client
    ml_client = get_ml_client()

    job_list = []
    print("JOBID\t\t\t\tNAME\t\tPARTITION\tSTATE\tTIME")
    for page in ml_client.jobs.list().by_page():
        for job in page:
            line = job.name
            if len(line) < 24:
                line += "\t"
            line += "\t" + job.display_name
            if len(line) < 49:
                line += "\t"
            line += "\t" + str(job.compute)
            line += "\t" + str(getattr(job, "status", "") or "")
            print(line)

