import os
import glob

python = '/home/chenjd/sort/virtualenv/tealeaf/venv/bin/python'
imagesplit = '/home/chenjd/web/TeaGVD/scripts/imagesplit.py'
imageleafmix = '/home/chenjd/web/TeaGVD/scripts/mix.py'
uploadspath = '/home/chenjd/web/TeaGVD/uploads/'

def oneImageLeaf(filename,file_url):
    os.system(python + ' ' + imagesplit + ' -i ' + uploadspath  + filename)
    splitimgs = glob.glob(uploadspath + "{}_R[0-9].png".format(filename.strip('.png')))
    splitimgs_url = []
    gray_url = []
    binary_url = []
    i = 0
    for img in splitimgs:
        i = i + 1
        splitimgs_url.append("{1}_R{0}.png".format(i, file_url.strip('.png')))
        gray_url.append("{1}_R{0}_gray.png".format(i, file_url.strip('.png')))
        binary_url.append("{1}_R{0}_binary.png".format(i, file_url.strip('.png')))
    print(splitimgs_url)
    value_arr = []
    f = os.popen(python + ' ' + imageleafmix + ' -i ' + uploadspath  + filename.strip('.png'))
    lines = f.read().split("\n")
    lines.pop()
    for line in lines:
        d = {}
        j = 0
        for ele in line.split("\t"):
            j = j + 1
            j_str = str(j)
            if j > 1:
                d[j_str] = ele
            else:
                d[j_str] = ele.split('/')[-1].strip('.png')
        value_arr.append(d)
    columns = [
            {
                    "field": "1",
                    "title": "编号",
            }, {
                    "field": "2",
                    "title": "周长",
            }, {
                    "field": "3",
                    "title": "面积",
            }, {
                    "field": "4",
                    "title": "长轴",
            }, {
                    "field": "5",
                    "title": "短轴",
            }, {
                    "field": "6",
                    "title": "长宽比",
            }, {
                    "field": "7",
                    "title": "范围",
            }, {
                    "field": "8",
                    "title": "坚实度",
            },{
                    "field": "9",
                    "title": "等效直径",
            }, {
                    "field": "10",
                    "title": "圆形度",
            }, {
                    "field": "11",
                    "title": "contrast_0",
            }, {
                    "field": "12",
                    "title": "dissimilarity_0",
            }, {
                    "field": "13",
                    "title": "homogeneity_0",
            }, {
                    "field": "14",
                    "title": "correlation_0",
            }, {
                    "field": "15",
                    "title": "ASM_0",
            }, {
                    "field": "16",
                    "title": "entropy_0",
            },{
                    "field": "17",
                    "title": "h_mean",
            }, {
                    "field": "18",
                    "title": "s_mean",
            }, {
                    "field": "19",
                    "title": "v_mean",
            }, {
                    "field": "20",
                    "title": "h_std",
            }, {
                    "field": "21",
                    "title": "s_std",
            }, {
                    "field": "22",
                    "title": "v_std",
            }, {
                    "field": "23",
                    "title": "h_skewness",
            }, {
                    "field": "24",
                    "title": "s_skewness",
            }, {
                    "field": "25",
                    "title": "v_skewness",
            }
        ]
    return splitimgs_url, gray_url, binary_url, value_arr, columns