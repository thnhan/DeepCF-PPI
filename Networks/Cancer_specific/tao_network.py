def get_protID_FASTA(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()
        protIDs = []
        if len(lines) > 0:
            for l in lines:
                if l.startswith('>'):
                    temp = l.split('|')
                    # print(temp)
                    protIDs.append(temp[2].split('_')[0])
            print("====== FILE is EMPTY =======")

    return protIDs


A_IDs = get_protID_FASTA('Cancer_specific_ProA.txt')
B_IDs = get_protID_FASTA('Cancer_specific_ProB.txt')
edges = [(A_IDs[i], B_IDs[i]) for i in range(len(A_IDs))]
print(edges)
