from point import Point
import core
import config

def main():

    globaldata = ["start"]
    wallpts, interiorpts, outerpts = 0,0,0
    wallptsidx, interiorptsidx, outerptsidx, table = [],[],[],[]

    file1 = open("preprocessorfile_normal.txt")
    data1 = file1.read()
    splitdata = data1.split("\n")
    splitdata = splitdata[:-1]

    defprimal = core.getInitialPrimitive()

    for _, itm in enumerate(splitdata):
        itmdata = itm.split(" ")
        itmdata.pop(-1)
        temp = Point(int(itmdata[0]), float(itmdata[1]), float(itmdata[2]), int(itmdata[3]), int(itmdata[4]), int(itmdata[5]), int(itmdata[6]), int(itmdata[19]), list(map(int,itmdata[20:])), 0, 1, defprimal, None, None, None, None, None, None, None, None, None, None, None, None, None)
        globaldata.append(temp)
        if int(itmdata[5]) == 0:
            wallpts += 1
            wallptsidx.append(int(itmdata[0]))
        elif int(itmdata[5]) == 1:
            interiorpts += 1
            interiorptsidx.append(int(itmdata[0]))
        elif int(itmdata[5]) == 2:
            outerpts += 1
            outerptsidx.append(int(itmdata[0]))
        table.append(int(itmdata[0]))

    
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

    for idx in table:
        connectivity = core.calculateConnectivity(globaldata, idx)
        globaldata[idx].setConnectivity(connectivity)

    for i in range(1, int(config.getConfig()["core"]["max_iters"]) + 1):
        core.fpi_solver(i, globaldata)

if __name__ == "__main__":
    main()
    