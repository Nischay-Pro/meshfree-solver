import networkx as nx
import argparse
import subprocess
from networkx.utils import cuthill_mckee_ordering

def main():

    G = nx.Graph()

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Grid File Input", type=str, default="partGridNew")
    parser.add_argument("-o", "--output", help="Grid File Output", type=str, default="partGridNew_Balanced")
    parser.add_argument("-l", "--legacy", help="Legacy Format", type=int, default=0)
    parser.add_argument("-u", "--heuristics", help="Heuristics", type=int, default=0)
    args = parser.parse_args()

    file_length = int(subprocess.run(['wc', '-l', args.input], stdout=subprocess.PIPE).stdout.decode('utf-8').split(" ")[0])
    G.add_nodes_from(range(1,file_length + 1))
    data = []
    data_new = []

    if args.legacy == 1:
        with open(args.input) as fileobject:
            for row, line in enumerate(fileobject):
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
        if args.heuristics == 0:
            result = list(cuthill_mckee_ordering(G))        
        elif args.heuristics == 1:
            result = list(cuthill_mckee_ordering(G, heuristic=smallest_degree))
        elif args.heuristics == 2:
            result = list(cuthill_mckee_ordering(G, heuristic=biggest_degree))
        else:
            exit()
        renumbering_matrix = list(zip(range(1, file_length + 1), result))
        for item in data:
            temp = [result[itm - 1] if isinstance(itm, int) else itm for itm in item]
            data_new.append(temp)
        renumbering_matrix = sorted(renumbering_matrix, key=lambda tup: tup[1])
        with open(args.output, "w+") as fileobject:
            for itm in renumbering_matrix:
                idx = itm[0]
                temp = data_new[idx - 1]
                temp = " ".join(list(map(str, temp)))
                fileobject.write(temp + "\n")
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