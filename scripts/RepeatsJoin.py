import sys, getopt
import ImageCharacters as ic

opts, args = getopt.getopt(sys.argv[1:],"a:b:c:t:o:",["afile=","bfile=","columns=","trait=","out="])
for opt, arg in opts:
    if opt in ("-a", "--afile"):
        afile = arg
    if opt in ("-b", "--bfile"):
        bfile = arg
    if opt in ("-c", "--columns"):
        col = arg
    if opt in ("-t", "--trait"):
        trait = arg
    if opt in ("-o", "--out"):
        output = arg

d = {}
rank = 1
with open(afile, 'r') as f:
    for line in f.readlines():
        line = line.strip("\n")
        if 'ID' in line:
            header = line.split(",")
            for i in range(1, len(header), 1):
                if header[i] == trait:
                    rank = i
        else:
            ele = line.split(",")
            d[ele[0]] = ele[rank]

print(col)
rank = 1
with open(bfile, 'r') as f:
    for line in f.readlines():
        line = line.strip("\n")
        if 'ID' in line:
            header = line.split(",")
            for i in range(1, len(header), 1):
                if header[i] == trait:
                    rank = i
        else:
            ele = line.split(",")
            if ele[0] in d:
                print(ele[0] + "," + d[ele[0]] + "," + ele[rank])

