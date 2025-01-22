# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 13:32:13 2021

@author: Chen

usage: python.exe ImageRecognitionThreshold.py -i GR2.png

"""
import sys, getopt
import cv2 as cv
import glob
import numpy as np

def threshTwoPeaks(image):
        histogram=np.bincount(image.ravel(),minlength = 256)
        maxLoc=np.where(histogram==np.max(histogram))
        firstPeak=maxLoc[0][0]#返回整数
        measureDists=np.zeros([256],np.float32)
        for k in range(256):
            measureDists[k]=pow(k-firstPeak,2)*histogram[k]#让原来的最高峰变为最低峰
        maxLoc2=np.where(measureDists==np.max(measureDists))
        secondPeak=maxLoc2[0][0]
        thresh=0
        if firstPeak>secondPeak:
            temp=histogram[int(secondPeak):int(firstPeak)]#求最高峰和第二高峰之间的直方图分布
            minLoc=np.where(temp==np.min(temp))
            thresh=secondPeak+minLoc[0][0]+1
        else:
            temp=histogram[int(firstPeak):int(secondPeak)]
            minLoc=np.where(temp==np.min(temp))
            thresh=firstPeak+minLoc[0][0]+1
        
        if thresh >40 or thresh<15:
            thresh = 25
        print(thresh)
        ret, threshImage_out = cv.threshold(image, thresh, 255, cv.THRESH_BINARY)  
        return (thresh,threshImage_out)
#images = glob.glob("*.png")
inputfile = ''
outputdir = ''
opts, args = getopt.getopt(sys.argv[1:],"hi:o:",["ifile=","odir="])
for opt, arg in opts:
    if opt in ("-i", "--ifile"):
        inputfile = arg
    elif opt in ("-o", "--odir"):
        outputdir = arg
images = []
images.append(inputfile)
kernal = np.ones((5,5),np.uint8)
for fname in images:
    #image = cv.imread(fname)
    image = cv.imdecode(np.fromfile(fname,dtype=np.uint8),1)#读取中文路径
    print(fname)
    #边缘填充
    img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur( img,(9,9),0)
    ret, thresh = threshTwoPeaks(img)

    closing = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernal)
    opening = cv.morphologyEx(closing, cv.MORPH_OPEN, kernal)
    
    contours, hierarchy = cv.findContours(opening, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    i=0
    for cnt in contours :
        A = cv.contourArea(cnt)
        #通过面积来将轮廓筛选
        if A > 3000 and A < 700000:
        # 找到边界坐标
            i = i + 1
            x, y, w, h = cv.boundingRect(cnt)  # 计算点集最外面的矩形边界
            (X,Y),(MA,ma),angle1 = cv.fitEllipse(cnt) #利用最小外接椭圆来计算角度
            image = cv.rectangle(image, (x, y), (x+w, y+h), (255, 255, 255), 4)
            image = cv.putText(image, str(i), (x, y - 20), cv.FONT_HERSHEY_COMPLEX, 4, (255, 255, 255))
#            cv.imencode('.jpg', roi1)[1].tofile('test.png')
    cv.imencode('.jpg', image)[1].tofile('{0}_marker.png'.format(fname.strip('.png')))

