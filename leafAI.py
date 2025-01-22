# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 10:58:34 2022

@author: Chen Jiedan
"""

import sys, getopt
import scripts.ImageCharacters as ic
import scripts.Plot as plot
import cv2 as cv
import numpy as np
import glob
import csv
import os

inputfile = ''
inputdir = ''
output = ''
hsvThresh = ''
opts, args = getopt.getopt(sys.argv[1:],"p:i:d:o:h:",["program=","infile=","indir=","out=","hsvThresh="])
for opt, arg in opts:
    if opt in ("-p", "--program"):
        program = arg
    if opt in ("-i", "--infile"):
        inputfile = arg
    if opt in ("-d", "--indir"):
        inputdir = arg
    if opt in ("-o", "--out"):
        output = arg
    if opt in ("-h", "--hsvThresh"):
        hsvThresh = arg

images = []
if inputdir != '':
    images = glob.glob(inputdir + "/*_R*[0-9].png")
if inputfile != '':
    images.append(inputfile)
if output == '':
    output = 'output'
if hsvThresh == '':
    hsvThresh = '0,160,0,40,170,255'

if program == 'shape':
    yc = []
    data = []
    for fname in images:
        try:
            img = cv.imdecode(np.fromfile(fname, dtype=np.uint8), 1)  # 读取中文路径
            gray,binary = ic.Preprocessing(img)
            cv.imencode('.jpg', gray)[1].tofile('{0}_gray.png'.format(fname.strip('.png')))
            cv.imencode('.jpg', binary)[1].tofile('{0}_binary.png'.format(fname.strip('.png')))
            sf = ic.ShapeFeatures(binary)
            name = os.path.basename(fname).strip('.png')
            sf.insert(0,name)
            data.append(sf)
        except:
            yc.append(fname)
    outname = output + '.shape.csv'
    ic.output_csv(outname,data,['ID','perimeter','area','length','width','aspectRatio','extent','solidity','equivalentDiameter','roundness'])
    # pp.output_csv(fs,['编号','周长','面积','长轴','短轴','长宽比','范围','坚实度','等效直径','圆形度'])
    # https://blog.csdn.net/qq_45194239/article/details/132189705
elif program == 'triHSVDot':
    for fname in images:
        img = cv.imdecode(np.fromfile(fname, dtype=np.uint8), 1)  # 读取中文路径
        x, y, z = ic.Bud_pixel(img)
        #figname = output + '.HSV.png'
        plot.Dot3CMatrix(x, y, z, 'h', 's', 'v', output + '.HSVdis.png')
elif program == 'triMask':
    for fname in images:
        img = cv.imdecode(np.fromfile(fname, dtype=np.uint8), 1)
        hmin,hmax,smin,smax,vmin,vmax = 0,160,0,40,170,255
        for i in range(40, 100, 10):
            for j in range(50, 140, 10):
                smax = i
                vmin = j
                mask_img = ic.Bud_triMask(img, hmin, smin, vmin, hmax, smax, vmax)
                bud,area,index =ic.Bud_index(img, hmin, smin, vmin, hmax, smax, vmax)
                image = cv.putText(mask_img, str("TI: {:.2f}".format(index)), (20, 30), cv.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255))
                cv.imencode('.jpg', mask_img)[1].tofile(output + '_mask_' + str(hmin) + '_' + str(hmax) + '_' + 
                                                        str(smin) + '_' + str(smax) + '_' + str(vmin) + '_' + str(vmax) + '.png')
elif program == 'triIndex':
    yc = []
    data = []
    for fname in images:
        try:
            img = cv.imdecode(np.fromfile(fname, dtype=np.uint8), 1)
            hmin,hmax,smin,smax,vmin,vmax = list(hsvThresh.split(","))
            ti =ic.Bud_index(img, int(hmin), int(smin), int(vmin), int(hmax), int(smax), int(vmax))
            name = os.path.basename(fname).strip('.png')
            ti.insert(0,name)
            data.append(ti)
        except:
            yc.append(fname)
    outname = output + '.trichome.csv'
    ic.output_csv(outname,data,['ID','trichome_pixel','bud_pixel','trichome_index'])
        #print(str(bud) + ';' + str(area) + ';' + str(index))
#        bud_pixel.insert(0, fname.strip('.png'))
 #       data.append(bud_pixel)
 #   outname = output + '.pixel.hsv.csv'
 #   ic.output_csv(outname,data,['ID','hsv'])

