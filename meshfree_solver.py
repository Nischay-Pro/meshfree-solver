from point import Point
import core
import config

def main():

    globaldata = ["start"]

    configData = config.getConfig()

    wallpts, interiorpts, outerpts = 0,0,0
    wallptsidx, interiorptsidx, outerptsidx, table = [],[],[],[]

    file1 = open("partGridNew")
    data1 = file1.read()
    splitdata = data1.split("\n")
    splitdata = splitdata[:-1]

    defprimal = core.getInitialPrimitive2(configData)

    for idx, itm in enumerate(splitdata):
        itmdata = itm.split(" ")
        temp = Point(int(itmdata[0]), float(itmdata[1]), float(itmdata[2]), 1, 1, int(itmdata[5]), int(itmdata[6]), int(itmdata[7]), list(map(int,itmdata[8:])), float(itmdata[3]), float(itmdata[4]), defprimal[idx], None, None, None, None, None, None, None, None, None, None, None, None, None)
        globaldata.append(temp)
        if int(itmdata[5]) == 1:
            wallpts += 1
            wallptsidx.append(int(itmdata[0]))
        elif int(itmdata[5]) == 2:
            interiorpts += 1
            interiorptsidx.append(int(itmdata[0]))
        elif int(itmdata[5]) == 3:
            outerpts += 1
            outerptsidx.append(int(itmdata[0]))
        table.append(int(itmdata[0]))

    
    # for idx in wallptsidx:
    #     currpt = globaldata[idx].getxy()
    #     leftpt = globaldata[idx].left
    #     leftpt = globaldata[leftpt].getxy()
    #     rightpt = globaldata[idx].right
    #     rightpt = globaldata[rightpt].getxy()
    #     normals = core.calculateNormals(leftpt, rightpt, currpt[0], currpt[1])
    #     globaldata[idx].setNormals(normals)

    # for idx in outerptsidx:
    #     currpt = globaldata[idx].getxy()
    #     leftpt = globaldata[idx].left
    #     leftpt = globaldata[leftpt].getxy()
    #     rightpt = globaldata[idx].right
    #     rightpt = globaldata[rightpt].getxy()
    #     normals = core.calculateNormals(leftpt, rightpt, currpt[0], currpt[1])
    #     globaldata[idx].setNormals(normals)

    for idx in table:
        connectivity = core.calculateConnectivity(globaldata, idx)
        globaldata[idx].setConnectivity(connectivity)

    res_old = 0

    for i in range(1, int(config.getConfig()["core"]["max_iters"]) + 1):
        res_old, globaldata = core.fpi_solver(i, globaldata, configData, wallptsidx, outerptsidx, interiorptsidx, res_old)

    for idx, itm in enumerate(globaldata):
        if idx > 0:
            primtowrite = globaldata[idx].prim
            with open("primvals.txt", "a") as the_file:
                for itm in primtowrite:
                    the_file.write(str(itm) + " ")
                the_file.write("\n")

if __name__ == "__main__":
    main()
    