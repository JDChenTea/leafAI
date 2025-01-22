import sys, getopt
import numpy as np
from scipy import stats as sts
from scipy.stats import pearsonr
import seaborn as sns
import Plot as plot
import pandas as pd

valuefile = ''
output = ''
opts, args = getopt.getopt(sys.argv[1:],"v:o:",["value=","out="])
for opt, arg in opts:
    if opt in ("-v", "--value"):
        valuefile = arg
    if opt in ("-o", "--out"):
        output = arg

data = []
header = []
xvalue = []
yvalue = []
with open(valuefile, 'r') as f:
    for line in f.readlines():
        line = line.strip("\n")
        if 'ID' in line:
            header = line.split(",")
            data  = pd.DataFrame(columns=[header[0],header[1],header[2]])
        else:
            ele = line.split(",")
            line = {header[0] : ele[0], header[1] : float(ele[1]), header[2] : float(ele[2])}
            xvalue.append(float(ele[1]))
            yvalue.append(float(ele[2]))
            data = data.append(line,ignore_index=True)

# 计算皮尔逊相关系数
correlation, p_value = pearsonr(xvalue, yvalue)
print(f"皮尔逊相关系数: {correlation}, p值: {p_value}")

plot.LinearRegMD(data, output, header[1], header[2])








