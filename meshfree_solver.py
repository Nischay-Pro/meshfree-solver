from point import Point
import core

def main():

    globaldata = []
    wallpts, interiorpts, outerpts = 0,0,0
    wallptsidx, interiorptsidx, outerptsidx = [],[],[]

    file1 = open("preprocessorfile_normal.txt")
    data1 = file1.read()
    splitdata = data1.split("\n")
    splitdata = splitdata[:-1]

    defprimal = core.getInitialPrimitive()

    print(defprimal)
    exit()

    for _, itm in enumerate(splitdata):
        itmdata = itm.split(" ")
        itmdata.pop(-1)
        temp = Point(itmdata[0], itmdata[1], itmdata[2], itmdata[3], itmdata[4], itmdata[5], itmdata[6], itmdata[19], itmdata[20:], 0, 0, None, None, defprimal, None, None, None, None, None, None, None, None, None, None, None)
        globaldata.append(temp)
        if int(itmdata[5]) == 0:
            wallpts += 1
            wallptsidx.append(itmdata[0])
        elif int(itmdata[5]) == 1:
            interiorpts += 1
            interiorptsidx.append(itmdata[0])
        elif int(itmdata[5]) == 2:
            outerpts += 1
            outerptsidx.append(itmdata[0])

    


if __name__ == "__main__":
    main()
    