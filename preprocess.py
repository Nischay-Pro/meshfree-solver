import argparse
import glob
import linecache
import subprocess as s
import os
import shutil
import fileinput
import in_place

def main():
    if not os.path.exists('temp'):
        os.makedirs('temp')
    else:
        shutil.rmtree('temp')
        os.makedirs('temp')
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--partition", help="Partition File Location", type=str, default=None)
    args = parser.parse_args()
    partition_file = args.partition
    order = []
    partition = []
    partition_org = []
    partition_file = partition_file.split("0")[0]
    for file in glob.glob(partition_file + "*"):
        order.append(file)
        line_counter = linecache.getline(file, 1)
        line_counter = list(map(int,line_counter.split(" ")))
        args = ['csplit', str(file), str(line_counter[1] + 2), '-f', 'temp/%s_' % os.path.basename(file), '-q']
        result = s.getoutput(" ".join(args))
        if len(result) == 0:
            partition.append('temp/%s_01' % os.path.basename(file))
            partition_org.append('temp/%s_00' % os.path.basename(file))
        else:
            print("Error")

    partition_original = {}
    partition_ghost = {}


    # for itm in partition:
    #     with open(itm) as the_file:
    #         key = int(os.path.basename(itm).split("_")[0].replace("partGrid",""))
    #         data = the_file.read().split("\n")
    #         data.pop(-1)
    #         for itm2 in data:
    #             itm2 = itm2.split(" ")
    #             itm2 = int(itm2[1])
    #             if key not in partition_ghost.keys():
    #                 partition_ghost[key] = [itm2]
    #             else:
    #                 partition_ghost[key].append(itm2)
    
    for itm in partition_org:
        with open(itm) as the_file:
            key = int(os.path.basename(itm).split("_")[0].replace("partGrid",""))
            data = the_file.read().split("\n")
            data.pop(-1)
            data.pop(0)
            for itm2 in data:
                itm2 = itm2.split(" ")
                itm2 = int(itm2[1])
                if key not in partition_original.keys():
                    partition_original[key] = [itm2]
                else:
                    partition_original[key].append(itm2)

    # assert len(partition_ghost) == len(partition_original)

    for itm in order:
        key = int(os.path.basename(itm).replace("partGrid",""))
        with in_place.InPlace(itm) as file:
            for idx, line in enumerate(file):
                if idx > 0:
                    splitdata = line.split()
                    if len(splitdata) < 9:
                        checker = int(splitdata[1])
                        for itm2 in partition_original.keys():
                            if itm2 != key:
                                if checker in partition_original[itm2]:
                                    line = line.replace("\n","") + "%s \n" %(itm2)
                file.write(line)

    shutil.rmtree('temp')


if __name__ == "__main__":
    main()