import argparse
import glob
import linecache
import subprocess as s
import os
import shutil

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
    partition_file = partition_file.split("0")[0]
    for file in glob.glob(partition_file + "*"):
        order.append(file)
        line_counter = linecache.getline(file, 1)
        line_counter = list(map(int,line_counter.split(" ")))
        args = ['csplit', str(file), str(line_counter[1] + 2), '-f', 'temp/%s_' % os.path.basename(file), '-q']
        result = s.getoutput(" ".join(args))
        if len(result) == 0:
            partition.append('temp/%s_01' % os.path.basename(file))
        else:
            print("Error")
    
    for itm in partition:
        with open(itm) as the_file:
            data = the_file.read().split("\n")
            data.pop(-1)
            for line in data:
                for itm2 in order:
                    if os.path.basename(itm2) in os.path.basename(itm):
                        pass
                    else:
                        if exists_in_file(itm2, line):
                            base = os.path.basename(itm2)
                            base = base.split("_")[0]
                            corebase = base
                            base = base.replace("partGrid", "")
                            base = int(base)
                            path = os.path.dirname(itm2)
                            base2 = os.path.basename(itm)
                            base2 = base2.split("_")[0]
                            replace_string(os.path.join(path, base2), line, line + str(base) + " ")
                            break

    shutil.rmtree('temp')

def replace_string(file, string1, string2):
    args = ["sed", "-i", "'s/%s/%s/g'" % (string1, string2), file]
    print(file, string1, string2)
    result = s.getoutput(" ".join(args))
    if len(result) > 0:
        return True
    else:
        return False

def exists_in_file(file, string):
    string = string.split(" ")
    string.pop(0)
    string.pop(0)
    string.pop(-1)
    string = " ".join(string)
    if string[0] != "-":
        args = ["grep", "'%s [0-9]\{1,20\} [0-9]\{1,20\}'" % string, file]
    else:
        args = ["grep", "'\%s [0-9]\{1,20\} [0-9]\{1,20\}'" % string, file]
    result = s.getoutput(" ".join(args))
    if len(result) > 0:
        return True
    else:
        return False

if __name__ == "__main__":
    main()