from point import Point
import core
import config
import argparse
from progress import printProgressBar
from tqdm import tqdm

def main():

    globaldata = ["start"]

    configData = config.getConfig()

    wallpts, interiorpts, outerpts = 0,0,0
    wallptsidx, interiorptsidx, outerptsidx, table = [],[],[],[]

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="Grid File Location", type=str, default="partGridNew")
    parser.add_argument("-t", "--thread", help="Thread Block Size", type=int, default=0)
    args = parser.parse_args()

    if not args.thread == 0:
        configData["core"]["blockGridX"] = args.thread

    file1 = open(args.file)
    print("Loading file: %s" % args.file)
    data1 = file1.read()
    splitdata = data1.split("\n")
    if len(splitdata[-1]) < 4:
        splitdata = splitdata[:-1]
    print("Getting Primitive Values Default")
    defprimal = core.getInitialPrimitive(configData)

    format = configData["core"]["format"]

    # Format 0: 6th Grid Format
    # Format 1: Old Format
    # Format 2: QuadTree Format

    if format == 2:
        splitdata.pop(0)

    print("Converting RAW dataset to Globaldata")
    for idx, itm in enumerate(tqdm(splitdata)):
        itmdata = itm.split(" ")[:-1]
        if format == 1:
            temp = Point(int(itmdata[0]), float(itmdata[1]), float(itmdata[2]), 1, 1, int(itmdata[5]), int(itmdata[6]), int(itmdata[8]), list(map(int,itmdata[9:])), float(itmdata[3]), float(itmdata[4]), defprimal, None, None, None, None, None, None, None, None, None, None, None, None, None, float(itmdata[7]))
        elif format == 2:
            temp = Point(idx + 1, float(itmdata[0]), float(itmdata[1]), int(itmdata[2]), int(itmdata[3]), int(itmdata[4]), int(itmdata[5]), int(itmdata[10]), list(map(int, itmdata[11:])), float(itmdata[6]), float(itmdata[7]), defprimal, None, None, None, None, None, None, None, None, None, None, None, None, None, float(itmdata[9]))
        else:
            temp = Point(int(itmdata[0]), float(itmdata[1]), float(itmdata[2]), int(itmdata[3]), int(itmdata[4]), int(itmdata[5]), int(itmdata[6]), int(itmdata[8]), list(map(int,itmdata[9:])), 1, 0, defprimal, None, None, None, None, None, None, None, None, None, None, None, None, None, float(itmdata[7]))
        globaldata.append(temp)
        if format == 0 or format == 1:
            if int(itmdata[5]) == configData["point"]["wall"]:
                wallpts += 1
                wallptsidx.append(int(itmdata[0]))
            elif int(itmdata[5]) == configData["point"]["interior"]:
                interiorpts += 1
                interiorptsidx.append(int(itmdata[0]))
            elif int(itmdata[5]) == configData["point"]["outer"]:
                outerpts += 1
                outerptsidx.append(int(itmdata[0]))
        else:
            if int(itmdata[4]) == configData["point"]["wall"]:
                wallpts += 1
                wallptsidx.append(idx + 1)
            elif int(itmdata[4]) == configData["point"]["interior"]:
                interiorpts += 1
                interiorptsidx.append(idx + 1)
            elif int(itmdata[4]) == configData["point"]["outer"]:
                outerpts += 1
                outerptsidx.append(idx + 1)
        table.append(idx + 1)

    if format == 0 or format == 2:
        for idx in wallptsidx:
            currpt = globaldata[idx].getxy()
            leftpt = globaldata[idx].left
            leftpt = globaldata[leftpt].getxy()
            rightpt = globaldata[idx].right
            rightpt = globaldata[rightpt].getxy()
            normals = core.calculateNormals(leftpt, rightpt, currpt[0], currpt[1])
            globaldata[idx].setNormals(normals)

        for idx in outerptsidx:
            currpt = globaldata[idx].getxy()
            leftpt = globaldata[idx].left
            leftpt = globaldata[leftpt].getxy()
            rightpt = globaldata[idx].right
            rightpt = globaldata[rightpt].getxy()
            normals = core.calculateNormals(leftpt, rightpt, currpt[0], currpt[1])
            globaldata[idx].setNormals(normals)

    print("Calculating Connectivity")
    for idx in table:
        connectivity = core.calculateConnectivity(globaldata, idx, configData)
        globaldata[idx].setConnectivity(connectivity)

    res_old = 0

    print("Starting FPI Solver")
    core.fpi_solver(config.getConfig()["core"]["max_iters"] + 1, globaldata, configData, wallptsidx, outerptsidx, interiorptsidx, res_old)

    print("Done")

if __name__ == "__main__":
    main()
    
