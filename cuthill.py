import networkx as nx
import argparse
import subprocess
from networkx.utils import cuthill_mckee_ordering, reverse_cuthill_mckee_ordering
from tqdm import tqdm

def main():

    G = nx.Graph()

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Grid File Input", type=str, default="partGridNew")
    parser.add_argument("-o", "--output", help="Grid File Output", type=str, default="partGridNew_Balanced")
    parser.add_argument("-l", "--legacy", help="Legacy Format", type=int, default=0)
    parser.add_argument("-u", "--heuristics", help="Heuristics", type=int, default=0)
    args = parser.parse_args()

    file_length = int(subprocess.run(['wc', '-l', args.input], stdout=subprocess.PIPE).stdout.decode('utf-8').split(" ")[0])
    if args.legacy == 1:
        G.add_nodes_from(range(1,file_length + 1))
    else:
        G.add_nodes_from(range(1,file_length))
    
    data = []
    data_new = []

    print("Loading File")

    if args.legacy == 1:
        with open(args.input) as fileobject:
            for row, line in enumerate(tqdm(fileobject, total=file_length)):
                line = line.replace("\n","")
                clean_line = line.split(" ")
                temp = []
                for idx, itm in enumerate(clean_line):
                    if idx == 0 or idx > 7:
                        itm = int(itm)
                    temp.append(itm)
                data.append(temp)
                connectivity_set = list(map(int, clean_line[8:]))
                connectivity_set = list(zip_with_scalar(connectivity_set, row + 1))
                G.add_edges_from(connectivity_set)

    else:
        with open(args.input) as fileobject:
            for row, line in enumerate(tqdm(fileobject, total=file_length)):
                if row > 0:
                    line = line.replace("\n","")
                    clean_line = line.split(" ")
                    clean_line.pop(-1)
                    data.append(clean_line)
                    connectivity_set = list(map(int, clean_line[11:]))
                    connectivity_set = list(zip_with_scalar(connectivity_set, row))
                    G.add_edges_from(connectivity_set)
    
    print("Applying Cuthill McKee")
    if args.heuristics == 0:
        result = list(cuthill_mckee_ordering(G))        
    elif args.heuristics == 1:
        result = list(cuthill_mckee_ordering(G, heuristic=smallest_degree))
    elif args.heuristics == 2:
        result = list(cuthill_mckee_ordering(G, heuristic=biggest_degree))
    else:
        exit()
    print("Converting Data Type")
    if args.legacy == 1:
        renumbering_matrix = list(zip(range(1, file_length + 1), result))
    else:
        renumbering_matrix = list(zip(range(1, file_length), result))
    for item in tqdm(data):
        flag = int(item[4])
        if flag == 0 or flag == 2:
            temp = [result[int(itm) - 1] if (idx == 2 or idx == 3 or idx > 10) else itm for idx, itm in enumerate(item)]
        else:
            temp = [result[int(itm) - 1] if (idx > 10) else itm for idx, itm in enumerate(item)]
        data_new.append(temp)
    renumbering_matrix = sorted(renumbering_matrix, key=lambda tup: tup[1])
    with open(args.output, "w+") as fileobject:
        if args.legacy == 1:
            fileobject.write("{}\n".format(file_length))
        else:
            fileobject.write("{}\n".format(file_length - 1))
        for itm in renumbering_matrix:
            idx = itm[0]
            temp = data_new[idx - 1]
            temp = " ".join(list(map(str, temp)))
            fileobject.write(temp + " \n")
    print("Done")

    # print(renumbering_matrix)

def zip_with_scalar(l, o):
    return ((i, o) for i in l)

def smallest_degree(G):
    node,_ = sorted(G.degree(), key = lambda x:x[1])[0]
    return node
    
def biggest_degree(G):
    node,_ = sorted(G.degree(), key = lambda x:x[1], reverse=True)[0]
    return node  

if __name__ == "__main__":
    main()