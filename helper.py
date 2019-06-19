def findMaxResidue(sum_res_sqr, limit=100):
    None

def printPrimitive(globaldata):
    with open("output.dat", "w+") as the_file:
        for itm in globaldata:
            primval = "{} {} {} {} {}".format(itm.localID, itm.prim[0], itm.prim[1], itm.prim[2], itm.prim[3])
            the_file.write("{}\n".format(primval))