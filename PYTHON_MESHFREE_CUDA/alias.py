import argparse
from tqdm import tqdm
import subprocess

def main():
    alias = {}
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Grid File Input", type=str, default="partGridNew")
    parser.add_argument("-o", "--output", help="Grid File Output", type=str, default="partGridNew_Balanced")
    parser.add_argument("-l", "--legacy", help="Legacy Format", type=int, default=0)
    args = parser.parse_args()
    data = []

    file_length = int(subprocess.run(['wc', '-l', args.input], stdout=subprocess.PIPE).stdout.decode('utf-8').split(" ")[0])

    print("Loading File")

    count = 0

    if args.legacy == 1:
        with open(args.input) as fileobject:
            for row, line in enumerate(tqdm(fileobject, total=file_length)):
                line = line.replace("\n","")
                clean_line = line.split(" ")
                temp = []
                ptidx = int(clean_line[0])
                for idx, itm in enumerate(clean_line):
                    if idx == 0 or idx > 7:
                        itm = int(itm)
                        if itm not in alias.keys():
                            count += 1
                            alias[itm] = count
                    temp.append(itm)
                data.append(temp)

    for itm in data:
        for idx, ix in enumerate(itm):
            if idx == 0 or idx > 7:
                itm[idx] = alias[ix]

    data = sorted(data, key=lambda tup: tup[0])

    print("Writing File")
    with open(args.output, "w+") as fileobject:
        for itm in tqdm(data):
            to_write = ""
            for idx, ix in enumerate(itm):
                if idx == 0:
                    to_write = to_write + "{}".format(str(ix))
                else:
                    to_write = to_write + " {}".format(str(ix))
            fileobject.write("{}\n".format(to_write))

    print("Done")

if __name__ == "__main__":
    main()