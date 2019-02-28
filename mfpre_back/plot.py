import glob
import numpy as np
import matplotlib.pyplot as plt

filelist=[]
for file in glob.glob("partGrid*"):
    filelist.append(file)

for fname in filelist:
    data = np.loadtxt(fname, delimiter='\n', dtype=str)
    num = int(data[0].split(' ')[1])
    X = []
    Y = []
    for i in range(1, num+1):
        X.append(float(data[i].split(' ')[2]))
        Y.append(float(data[i].split(' ')[3]))
    plt.scatter(X, Y, s=1.5,marker=".")

plt.show()
