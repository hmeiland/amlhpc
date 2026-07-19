def sinfo(vargs=None):
    from ..jobcontrol import get_ml_client
    ml_client = get_ml_client()

    sinfo_list = ml_client.compute.list()

    print("PARTITION\tAVAIL\tVM_SIZE\t\t\tNODES\tSTATE")
    for i in sinfo_list:
        line = i.name
        if len(line) < 8:
            line += "\t"
        try:
            line += "\tUP\t" + i.size + "\t"
        except:
            line += "\tUP\t" + "unknown" + "\t\t"
        if len(line.expandtabs()) < 41:
            line += "\t"
        try:
            line += str(i.max_instances) + "\t"
        except:
            line += "unknown" + "\t"
        try:
            line += str(i.state)
        except:
            line += "unknown"
        print(line)
