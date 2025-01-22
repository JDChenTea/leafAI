import sys, getopt
import numpy as np
from scipy import stats as sts
import ImageCharacters as ic


valuefile = ''
groupfile = ''
output = ''
opts, args = getopt.getopt(sys.argv[1:],"v:g:o:",["value=","group=","out="])
for opt, arg in opts:
    if opt in ("-v", "--value"):
        valuefile = arg
    if opt in ("-g", "--group"):
        groupfile = arg
    if opt in ("-o", "--out"):
        output = arg

group_d = {}
tmp_d = {}
with open(groupfile, 'r') as f:
    for line in f.readlines():
        line = line.strip("\n")
        ele = line.split(",")
        group_d[ele[0]] = ele[1]
        tmp_d[ele[1]] = ""

value_d = {}
header = []
with open(valuefile, 'r') as f:
    for line in f.readlines():
        line = line.strip("\n")
        if 'ID' in line:
            header = line.split(",")
 #           for i in range(1, len(header), 1):
 #               value_d[header[i]] = tmp_d
        else:
            ele = line.split(",")
            for i in range(1, len(ele), 1):
 #               print(i)
 #               print(header[i])
 #               print(value_d[header[i]][group_d[ele[0]]])
                if header[i] in value_d:
                    if group_d[ele[0]] in value_d[header[i]]:
                        value_d[header[i]][group_d[ele[0]]] = value_d[header[i]][group_d[ele[0]]] + ";" + ele[i]
                    else:
                        d = {group_d[ele[0]]: ele[i]}
                        value_d[header[i]].update(d)
                else:
                    d = {group_d[ele[0]]: ele[i]}
                    value_d[header[i]] =  d
                    

data = []
for key2 in tmp_d:
    mean_arr = []
    std_arr = []
    min_arr = []
    max_arr = []
    for i in range(1, len(header), 1):
#        print(header[i])
#        print(value_d[header[i]][key2])
        value_arr = list(float(x) for x in value_d[header[i]][key2].split(";"))
        repeats = len(value_arr)
#        print(value_arr)
        mean = str("{:.2f}".format(np.mean(value_arr)))
        mean_arr.append(mean)
        minv = str("{:.2f}".format(np.min(value_arr)))
        min_arr.append(minv)
        maxv = str("{:.2f}".format(np.max(value_arr)))
        max_arr.append(maxv)
        std = str("{:.2f}".format(sts.tstd(value_arr,ddof = 1)))
        std_arr.append(std)
    out_arr = [key2] + mean_arr + min_arr + max_arr + list(str(x) for x in std_arr) + [str(repeats)]
#    print(out_arr)
    data.append(out_arr)

outheader =['ID']
for i in range(1, len(header), 1):
    tmph = header[i] + '_mean'
    outheader.append(tmph)

for i in range(1, len(header), 1):
    tmph = header[i] + '_min'
    outheader.append(tmph)

for i in range(1, len(header), 1):
    tmph = header[i] + '_max'
    outheader.append(tmph)

for i in range(1, len(header), 1):
    tmph = header[i] + '_std'
    outheader.append(tmph)

outheader.append('repeats')

ic.output_csv(output,data,outheader)
#ic.output_csv(output,data,['ID','trichome_pixel_mean','bud_pixel_mean','trichome_index_mean','trichome_pixel_min','bud_pixel_min','trichome_index_min','trichome_pixel_max','bud_pixel_max','trichome_index_max','trichome_pixel_std','bud_pixel_std','trichome_index_std','repeats'])








