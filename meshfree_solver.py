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
    args = parser.parse_args()

    file1 = open(args.file)
    print("Loading file: %s" % args.file)
    data1 = file1.read()
    splitdata = data1.split("\n")
    splitdata = splitdata[1:-1]

    print("Getting Primitive Values Default")
    defprimal = core.getInitialPrimitive(configData)

    original_format = configData["core"]["format"]
    if original_format == 0:
        original_format = True
    else:
        original_format = False

    print("Converting RAW dataset to Globaldata")
    for idx, itm in enumerate(tqdm(splitdata)):
        itmdata = itm.split(" ")[1:-1]
        if not original_format:
            temp = Point(int(itmdata[0]), float(itmdata[1]), float(itmdata[2]), 1, 1, int(itmdata[5]), int(itmdata[6]), int(itmdata[8]), list(map(int,itmdata[9:])), float(itmdata[3]), float(itmdata[4]), defprimal, None, None, None, None, None, None, None, None, None, None, None, None, None, float(itmdata[7]), None, None)
        else:
            temp = Point(int(itmdata[0]), float(itmdata[1]), float(itmdata[2]), int(itmdata[3]), int(itmdata[4]), int(itmdata[5]), int(itmdata[6]), int(itmdata[7]), list(map(int,itmdata[8:])), 1, 0, defprimal, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None)
        globaldata.append(temp)
        if int(itmdata[5]) == configData["point"]["wall"]:
            wallpts += 1
            wallptsidx.append(int(itmdata[0]))
        elif int(itmdata[5]) == configData["point"]["interior"]:
            interiorpts += 1
            interiorptsidx.append(int(itmdata[0]))
        elif int(itmdata[5]) == configData["point"]["outer"]:
            outerpts += 1
            outerptsidx.append(int(itmdata[0]))
        table.append(int(itmdata[0]))

    if original_format:
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
    # for idx, itm in enumerate(globaldata):
    #     if idx > 0:
    #         primtowrite = globaldata[idx].prim
    #         with open("primvals.txt", "a") as the_file:
    #             for itm in primtowrite:
    #                 the_file.write(str(itm) + " ")
    #             the_file.write("\n")

if __name__ == "__main__":
    main()
    