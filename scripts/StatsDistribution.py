import sys, getopt
import numpy as np
from scipy import stats as sts
import seaborn as sns
import Plot as plot

opts, args = getopt.getopt(sys.argv[1:],"v:t:o:",["value=","trait=","out="])
for opt, arg in opts:
    if opt in ("-v", "--value"):
        valuefile = arg
    if opt in ("-t", "--trait"):
        trait = arg
    if opt in ("-o", "--out"):
        output = arg

data = []
rank = 1
with open(valuefile, 'r') as f:
    for line in f.readlines():
        line = line.strip("\n")
        if 'ID' in line:
            header = line.split(",")
            for i in range(1, len(header), 1):
                if header[i] == trait:
                    rank = i
        else:
            ele = line.split(",")
            data.append(float(ele[rank]))

print("Minimum:\t{:.2f}".format(np.min(data)))
print("Maximum:\t{:.2f}".format(np.max(data)))
print("Average:\t{:.2f}".format(np.mean(data)))
print("Standard Deviation:\t{:.2f}".format(sts.tstd(data,ddof = 1)))
print("Coefficient of Variation:\t{:.2f}".format(sts.tstd(data)/sts.tmean(data)))
plot.HistPlot(data, output)