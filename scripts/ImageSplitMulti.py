# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 13:32:13 2021

@author: Chen

usage: python.exe ImageSplitMulti.py -i data\
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

opts, args = getopt.getopt(sys.argv[1:],"hi:o:",["ifile=","odir="])
for opt, arg in opts:
    if opt in ("-i", "--ifile"):
        inputdir = arg
    elif opt in ("-o", "--odir"):
        outputdir = arg
images = glob.glob(inputdir + "*.png")

kernal = np.ones((5,5),np.uint8)
for fname in images:
    #image = cv.imread(fname)
    image = cv.imdecode(np.fromfile(fname,dtype=np.uint8),1)#读取中文路径
    print(fname)
    #边缘填充
    filling = cv.copyMakeBorder(image,500,500,500,500, cv.BORDER_CONSTANT,value=[0,0,0])
    result3 = filling.copy()
    img = cv.cvtColor(filling, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur( img,(9,9),0)
    ret, thresh = threshTwoPeaks(img)

    closing = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernal)
    opening = cv.morphologyEx(closing, cv.MORPH_OPEN, kernal)
    
    contours, hierarchy = cv.findContours(opening, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    i=0
    for cnt in contours :
        A = cv.contourArea(cnt)
        #通过面积来将轮廓筛选
        if A > 1000 and A < 7000000:
        # 找到边界坐标
            i = i + 1
            x, y, w, h = cv.boundingRect(cnt)  # 计算点集最外面的矩形边界
            (X,Y),(MA,ma),angle1 = cv.fitEllipse(cnt) #利用最小外接椭圆来计算角度

            roi1 = cv.rectangle(filling, (x, y), (x+w, y+h), (0, 0, 255), 5)
            #center = (int(abs(X)),int(abs(Y)))
            center = (int(abs(x+w/2)),int(abs(y+h/2)))
            #防止沿X轴计算角度或者斜向角度
            if 45<angle1<120:
                angle1 = angle1 + 90
            if 120<angle1<150:
                angle1 = angle1 + 45

            #print(x+w/2,y+h/2)
            angle2 = 180 - angle1
            if abs(angle1) < 45:
                M = cv.getRotationMatrix2D((center),angle1,1) 
            else:
                M = cv.getRotationMatrix2D((center),-angle2,1) 
            result2 = cv.warpAffine(result3, M, (filling.shape[1], filling.shape[0]),borderValue=(0,0,0))
            roi = result2[y-50:(y+h)+50,x-50:(x+w)+50]
            #roi = result2[y:(y+h),x:(x+w)]
            #cv.imwrite('F:/python_work/image_process/cut/result/bud/{1}_R{0}.png'
                       #.format(i,fname.strip('.png')),roi)
            #存储中文路径图片
            cv.imencode('.jpg', roi)[1].tofile('{1}_R{0}.png'.format(i,fname.strip('.png')))
