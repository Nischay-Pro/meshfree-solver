from point import Point
import core
import config
import argparse
from tqdm import tqdm

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
    # h5file = HDFStore(args.file)
    # for itm in h5file.walk():
    #     partitions = len(itm[1])
    #     break
    # print("Detected {} partition(s)".format(partitions))
    # print("Getting Primitive Values Default")
    # defprimal = core.getInitialPrimitive(configData)
    # for i in trange(1, partitions + 1):
    #     localData = h5file.get_node("/{}/{}".format(str(i), "local"))
    #     for itm in tqdm(localData):
    #         idx = int(itm[0])
    #         x = itm[1]
    #         y = itm[2]
    #         nx = 1
    #         ny = 0
    #         min_dist = itm[5]
    #         left = int(itm[6])
    #         right = int(itm[7])
    #         qt_depth = int(itm[8])
    #         flag_1 = int(itm[9])
    #         flag_2 = int(itm[10])
    #         nbhs = int(itm[11])
    #         conn = tuple(map(int, itm[12:12+nbhs]))
    #         temp = Point(idx, x, y, left, right, flag_1, flag_2, nbhs, conn, nx, ny, defprimal, None, None, None, None, None, None, None, None, None, None, None, None, None, min_dist, qt_depth)
    #         globaldata[idx] = temp
    #         if flag_1 == configData["point"]["wall"]:
    #             wallpts += 1
    #             wallptsidx.append(idx)
    #         elif flag_1 == configData["point"]["interior"]:
    #             interiorpts += 1
    #             interiorptsidx.append(idx)
    #         elif flag_1 == configData["point"]["outer"]:
    #             outerpts += 1
    #             outerptsidx.append(idx)

    # h5file.close()

    with open(args.file, "r") as inputFile:
        defprimal = core.getInitialPrimitive(configData)
        expected_points = int(inputFile.readline().split()[0])
        for idx, line in enumerate(tqdm(inputFile, total=expected_points)):
            line = line.split()
            if line == "":
                continue
            x = float(line[0])
            y = float(line[1])
            nx = 1
            ny = 0
            left = int(line[2])
            right = int(line[3])
            flag_1 = int(line[4])
            flag_2 = int(line[5])
            vor_area = float(line[6])
            min_dist = float(line[7])
            nbhs = int(line[8])
            conn = tuple(map(int, line[9:]))
            assert len(conn) == nbhs, f"Number of neighbors does not match for point {idx + 1}"
            globaldata[idx + 1] = Point(idx + 1, x, y, left, right, flag_1, flag_2, nbhs, conn, nx, ny, defprimal, min_dist)
            if flag_1 == configData["point"]["wall"]:
                wallpts += 1
                wallptsidx.append(idx + 1)
            elif flag_1 == configData["point"]["interior"]:
                interiorpts += 1
                interiorptsidx.append(idx + 1)
            elif flag_1 == configData["point"]["outer"]:
                outerpts += 1
                outerptsidx.append(idx + 1)
    
    assert len(globaldata) == expected_points, f"Number of points does not match for file {args.file}"
    print("Loaded {} points".format(len(globaldata)))

    for idx in (wallptsidx + outerptsidx):
        currpt = globaldata[idx].getxy()
        leftpt = globaldata[idx].left
        leftpt = globaldata[leftpt].getxy()
        rightpt = globaldata[idx].right
        rightpt = globaldata[rightpt].getxy()
        normals = core.calculateNormals(leftpt, rightpt, currpt[0], currpt[1])
        globaldata[idx].setNormals(normals)

    # for idx in outerptsidx:
    #     currpt = globaldata[idx].getxy()
    #     leftpt = globaldata[idx].left
    #     leftpt = globaldata[leftpt].getxy()
    #     rightpt = globaldata[idx].right
    #     rightpt = globaldata[rightpt].getxy()
    #     normals = core.calculateNormals(leftpt, rightpt, currpt[0], currpt[1])
    #     globaldata[idx].setNormals(normals)

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
    
