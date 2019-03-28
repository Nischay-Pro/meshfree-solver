from point import Point
import core
import config
import argparse
from progress import printProgressBar
import numpy as np
import tracemalloc
try:
    from mpi4py import MPI
    import dill
    import msgpack
    MPI_CAPABLE = True
    MPI.pickle.__init__(msgpack.dumps, msgpack.loads)
except ImportError:
    MPI_CAPABLE = False

def main():

    if MPI_CAPABLE:
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()

    globaldata = ["start"]

    configData = config.getConfig()

    original_format = configData["core"]["format"]
    if original_format == 0:
        original_format = True
    else:
        original_format = False

    wallpts, interiorpts, outerpts = 0,0,0
    wallptsidx, interiorptsidx, outerptsidx, table = [],[],[],[]


    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="Grid File Location", type=str, default="partGridNew")
    parser.add_argument("-p", "--partition", help="Partition File Location", type=str, default=None)
    MPI_MODE = False
    args = parser.parse_args()

    if MPI_CAPABLE and args.partition != None:
        MPI_MODE = True

    if not MPI_MODE:
        file1 = open(args.file)
        print("Loading file: %s" % args.file)
        data1 = file1.read()
        splitdata = data1.split("\n")
        splitdata = splitdata[:-1]

        print("Getting Primitive Values Default")
        defprimal = core.getInitialPrimitive(configData)

        print("Converting RAW dataset to Globaldata")
        for idx, itm in enumerate(splitdata):
            # printProgressBar(
            #     idx, len(splitdata) - 1, prefix="Progress:", suffix="Complete", length=50
            # )
            itmdata = itm.split(" ")
            if not original_format:
                temp = Point(int(itmdata[0]), float(itmdata[1]), float(itmdata[2]), 1, 1, int(itmdata[5]), int(itmdata[6]), int(itmdata[7]), list(map(int,itmdata[8:])), float(itmdata[3]), float(itmdata[4]), defprimal, None, None, None, None, None, None, None, None, None, None, None, None, None)
            else:
                temp = Point(int(itmdata[0]), float(itmdata[1]), float(itmdata[2]), int(itmdata[3]), int(itmdata[4]), int(itmdata[5]), int(itmdata[6]), int(itmdata[7]), list(map(int,itmdata[8:])), 1, 0, defprimal, None, None, None, None, None, None, None, None, None, None, None, None, None)
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

        print("Calculating Normals")

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

    else:
        if comm.rank == 0:
            print("Running in MPI mode")
            print("Running with %s cores" % comm.size)
        globaldata_ghost = {}
        globaldata_local = {}
        globaldata_table = {}
        total_local_points = 0
        total_ghost_points = 0
        curr_local = 0
        wallptsidx, interiorptsidx, outerptsidx = [],[],[]
        total_parts = str(size)
        padding_length = len(total_parts)
        if comm.rank == 0:
            print("Getting Primitive Values Default")
        defprimal = core.getInitialPrimitive(configData)
        if padding_length == 1:
            padding_length += 1
        partition_file = args.partition
        partition_file = partition_file.split("0")[0]
        file_path = args.partition + "0" + str(rank)
        if comm.rank == 0:
            print("Loading partition: %s" % args.partition)
        with open(file_path) as fileobject:
            for row, line in enumerate(fileobject):
                if row == 0:
                    foreigner = line
                    foreigner = tuple(map(int, foreigner.split(" ")))
                    total_local_points = foreigner[1]
                    total_ghost_points = foreigner[2]
                if row > 0:
                    temp = line.split(" ")
                    idx = int(temp[0])
                    temp.pop(-1)
                    if curr_local != total_local_points:
                        # Local Points
                        if original_format:
                            globaldata_local[idx] = Point(idx, float(temp[2]), float(temp[3]), int(temp[4]), int(temp[5]), int(temp[6]), int(temp[7]), int(temp[8]), tuple(map(int,temp[9:])), 1, 0, defprimal, None, None, None, None, None, None, None, None, None, None, None, None, None, foreign=False, globalID=int(temp[1]))
                        else:
                            globaldata_local[idx] = Point(idx, float(temp[2]), float(temp[3]), 1, 1, int(temp[6]), int(temp[7]), int(temp[8]), tuple(map(int,temp[9:])), float(temp[4]), float(temp[5]), defprimal, None, None, None, None, None, None, None, None, None, None, None, None, None, foreign=False, globalID=int(temp[1]))
                        globaldata_table[int(temp[1])] = idx
                        if int(temp[6]) == configData["point"]["wall"]:
                            wallptsidx.append(idx)
                        elif int(temp[6]) == configData["point"]["interior"]:
                            interiorptsidx.append(idx)
                        elif int(temp[6]) == configData["point"]["outer"]:
                            outerptsidx.append(idx)
                        curr_local += 1
                    else:
                        # Ghost Points
                        globaldata_ghost[idx] = Point(idx, float(temp[2]), float(temp[3]), None, None, None, None, None, None, None, None, defprimal, None, None, None, None, None, None, None, None, None, None, None, None, None, foreign=True, foreign_core=int(temp[4]), globalID=int(temp[1]))
                        globaldata_table[int(temp[1])] = idx

        assert len(globaldata_ghost) == total_ghost_points, "Invalid number of Ghost Points"
        assert len(globaldata_local) == total_local_points, "Invalid number of Local Points"

        if comm.rank == 0 and original_format:
            print("Calculating Normals")

            for idx in globaldata_local.keys():
                if globaldata_local[idx].flag_1 == configData["point"]["wall"] or globaldata_local[idx].flag_1 == configData["point"]["outer"]:
                    currpt = globaldata_local[idx].getxy()

                    leftpt = globaldata_local[idx].left
                    if leftpt in globaldata_local.keys():
                        leftpt = globaldata_local[leftpt].getxy()
                    else:
                        leftpt = globaldata_ghost[leftpt].getxy()

                    rightpt = globaldata_local[idx].right
                    if rightpt in globaldata_local.keys():
                        rightpt = globaldata_local[rightpt].getxy()
                    else:
                        rightpt = globaldata_ghost[rightpt].getxy()

                    normals = core.calculateNormals(leftpt, rightpt, currpt[0], currpt[1])
                    globaldata_local[idx].setNormals(normals)

        if comm.rank == 0:
            print("Calculating Connectivity")
        for idx in globaldata_local.keys():
            connectivity = core.calculateConnectivityMPI(globaldata_local, idx, configData, globaldata_ghost)
            globaldata_local[idx].setConnectivity(connectivity)

        res_old = 0

        if rank == 0:
            print("Starting FPI Solver")
        tracemalloc.start()
        core.fpi_solver_mpi(config.getConfig()["core"]["max_iters"] + 1, globaldata_local, configData, globaldata_ghost, res_old, wallptsidx, outerptsidx, interiorptsidx, comm, globaldata_table)
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        if rank == 0:
            print("[ Top 10 ]")
            for stat in top_stats[:10]:
                print(stat)

if __name__ == "__main__":
    main()
    