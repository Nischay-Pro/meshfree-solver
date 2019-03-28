import matplotlib.pyplot as plt
import re

def main():
    filedata = open("cp-file")
    filedata = filedata.read()
    filedata = filedata.split("\n")
    filedata.pop(-1)
    data = {}
    data_vals = {}
    for itm in filedata:
        itm = re.sub(' +', ' ', itm)
        itm = itm.strip()
        temp = itm.split(" ")
        if int(temp[0]) not in data.keys():
            idx = int(temp[0])
            data[idx] = []
            data_vals[idx] = []
            data[idx].append(float(temp[1]))
            data_vals[idx].append(float(temp[2]))
        else:
            idx = int(temp[0])
            data[idx].append(float(temp[1]))
            data_vals[idx].append(float(temp[2]))
    
    print("Plotting %s files" % len(data.keys()))
    for i in data.keys():
        fig = plt.figure()
        plt.plot(data[i], data_vals[i], 'r')
        plt.xlabel('Distance')
        plt.ylabel('Cp')
        plt.title('Cp plot for shape %s' % i)
        fig.savefig('cpplot_%s.pdf' % i)

if __name__ == "__main__":
    main()
