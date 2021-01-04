from point import Point
import core
import config
import argparse
from progress import printProgressBar
from tqdm import tqdm, trange
import output
import h5py
import math

def main():

    globaldata = {}

    configData = config.getConfig()

    wallpts, interiorpts, outerpts = 0,0,0
    wallptsidx, interiorptsidx, outerptsidx = [],[],[]

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="Grid File Location", type=str, default="partGridNew")
    parser.add_argument("-t", "--thread", help="Thread Block Size", type=int, default=0)
    parser.add_argument("-i", "--inner", help="Inner Loop Iterations", type=int, default=0)
    args = parser.parse_args()

    if not args.thread == 0:
        configData["core"]["blockGridX"] = args.thread

    if not args.inner == 0:
        configData["core"]["inner"] = args.inner

    print("Loading file: %s" % args.file)
    h5file = h5py.File(args.file, "r")
    partitions = len(h5file.keys())
    print("Detected {} partition(s)".format(partitions))
    print("Getting Primitive Values Default")
    defprimal = core.getInitialPrimitive(configData)
    for i in trange(1, partitions + 1):
        localData = h5file.get("{}/{}".format(str(i), "local"))
        localpts = localData.shape[0]
        for itm in localData:
            idx = int(itm[0])
            x = float(itm[1])
            y = float(itm[2])
            nx = 1
            ny = 0
            min_dist = float(itm[5])
            left = int(itm[6])
            right = int(itm[7])
            qt_depth = int(itm[8])
            flag_1 = int(itm[9])
            flag_2 = int(itm[10])
            nbhs = int(itm[11])
            conn = tuple(map(int, itm[12:12+nbhs]))
            temp = Point(idx, x, y, left, right, flag_1, flag_2, nbhs, conn, nx, ny, defprimal, None, None, None, None, None, None, None, None, None, None, None, None, None, min_dist, qt_depth)
            globaldata[idx] = temp
            if flag_1 == configData["point"]["wall"]:
                wallpts += 1
                wallptsidx.append(idx)
            elif flag_1 == configData["point"]["interior"]:
                interiorpts += 1
                interiorptsidx.append(idx)
            elif flag_1 == configData["point"]["outer"]:
                outerpts += 1
                outerptsidx.append(idx)

    for itm in globaldata.keys():
        if globaldata[itm].checkConnectivity():
            print(idx + 1)

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
    for idx in range(1, len(globaldata) + 1):
        connectivity = core.calculateConnectivity(globaldata, idx, configData)
        globaldata[idx].setConnectivity(connectivity)

    res_old = 0

    globaldata[0] = "Dummy"

    print("Starting FPI Solver")
    _, globaldata = core.fpi_solver(config.getConfig()["core"]["max_iters"] + 1, globaldata, configData, wallptsidx, outerptsidx, interiorptsidx, res_old)

    print("Done")

if __name__ == "__main__":
    main()
    
