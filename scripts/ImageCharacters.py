# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 09:27:32 2021

@author: Chen Qiyu
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 13:51:36 2021

@author: Chen Qiyu
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 14:41:58 2021

@author: Chen Qiyu
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 10:50:11 2022

@author: Chen
"""
import cv2 as cv
import numpy as np
import csv
import skimage.feature as skf
#from skimage.feature import greycomatrix, greycoprops
import collections.abc
from skimage.draw import polygon
from skimage import draw
import pandas as pd

dpi = 600
dpi2cm = 0.3937008 * dpi

kernal = np.ones((5,5),np.uint8)
def Maxgray(img):
    """最大值灰度法"""
    #获取图像高度和宽度
    height = img.shape[0]
    width = img.shape[1]
     
    #创建一幅图像
    grayimg = np.zeros((height, width, 3), np.uint8)
     
    #图像最大值灰度处理
    for i in range(height):
        for j in range(width):
            #获取图像R G B最大值
            gray = max(img[i,j][0], img[i,j][1], img[i,j][2])
            #灰度图像素赋值 gray=max(R,G,B)
            grayimg[i,j] = np.uint8(gray)
    return grayimg
def Preprocessing(image):
    """图像预处理，得到灰度图像和二值化图像"""
    
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray,(9,9),0)
    
    binary = ThreshTwoPeaks(blur)

    closing = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernal)
    opening = cv.morphologyEx(closing, cv.MORPH_OPEN, kernal)
   
    return gray, opening
def ThreshTwoPeaks(gray):
    """双峰阈值法"""
    histogram=np.bincount(gray.ravel(),minlength = 256)
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
#    print(thresh)
    if thresh >40:
        thresh = 50
#    print(thresh)
    ret, threshImage_out = cv.threshold(gray, thresh, 255, cv.THRESH_BINARY)  
    return (threshImage_out)
def Max_area_cnt(contours):
    """筛选出面积最大的轮廓
    3.0更新
    """ 
    areas = []
    for cnt in contours:
        A = cv.contourArea(cnt)
        areas.append(A)
    max_id = areas.index(max(areas))
    cnt = contours[max_id]
    return cnt
def ShapeFeatures(binary):
    """形态特征提取"""
    contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)  
    cnt = Max_area_cnt(contours)

    #周长
    perimeter = cv.arcLength(cnt,True)
    perimeter_cm = round(perimeter/dpi2cm,2)
    
    #面积
    area = cv.contourArea(cnt)
    area_cm = round(area/dpi2cm/dpi2cm,2)
    
    #长轴短轴
    x,y,w,h = cv.boundingRect(cnt)
    long_axis = round(h/dpi2cm,2)
    minor_axis =round(w/dpi2cm,2)
    #长宽比
    aspect_ratio = round(float(w)/h,2)
    
    #范围 范围是轮廓区域与边界矩形区域的比值。
    area = cv.contourArea(cnt)
    x,y,w,h = cv.boundingRect(cnt)
    rect_area = w*h
    extent = round(float(area)/rect_area,2)

    #坚实度 坚实度是目标面积与其凸包面积之比。
    hull = cv.convexHull(cnt)
    hull_area = cv.contourArea(hull)
    solidity = round(float(area)/hull_area,2)
    
    #等效直径 等效直径是面积与轮廓面积相同的圆的直径。
    area = cv.contourArea(cnt)
    equi_diameter = np.sqrt(4*area/np.pi)
    #equi_diameter_cm = "{:.2f}".format(equi_diameter/118)
    equi_diameter_cm = round(equi_diameter/dpi2cm,2)
    
    #圆形度e为1时，图形即为圆形；e越小，图形越不规律，与圆形的差距越大。e=（4π*面积）/（周长*周长）
    e = 4*np.pi*area/dpi2cm/dpi2cm/perimeter_cm/perimeter_cm
    #e = "{:.2f}".format(1/e)
    e = round(1/e,2)
 
    list = [perimeter_cm,area_cm,long_axis,minor_axis,aspect_ratio,
                 extent,solidity,equi_diameter_cm,e]
    
    return list
 
def Texture_feature(gray):
    """纹理特征提取
     names = ['contrast_0','contrast_45','contrast_90','contrast_135', 
           'dissimilarity_0','dissimilarity_45','dissimilarity_90','dissimilarity_135',
                 'homogeneity_0', 'homogeneity_45','homogeneity_90','homogeneity_135',
                 'correlation_0', 'correlation_45', 'correlation_90', 'correlation_135',
                 'ASM_0', 'ASM_45', 'ASM_90', 'ASM_135',
                 'entropy_0','entropy_45','entropy_90','entropy_135']
     提取特征：对比度，非相似性，同质性，相关性，角二阶矩，熵"""
    TF = []
    TFs = []
    # 得到共生矩阵，参数：图像矩阵，距离，方向，灰度级别，是否对称，是否标准化
    glcm = skf.greycomatrix(
        gray, [
            1], [
            0, #np.pi / 4, np.pi / 2, np.pi * 3 / 4
            ], 256, symmetric=True, normed=True)
    
    for prop in ['contrast', 'dissimilarity',
                 'homogeneity', 'correlation', 'ASM','entropy']:
        temp = skf.greycoprops(glcm, prop)
        temp = flatten(temp)
        TF.extend(temp)
    for n in TF:
        #n = "{:.3f}".format(n)
        n = round(n,3)
        TFs.extend([n])
        
    return TFs
def Area_extract(image):
    """提取目标区域像素点"""
    leafs =[]
    img_o = image.copy()
    gray,opening = Preprocessing(image)
    
    contours, hierarchy = cv.findContours(opening, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    cols = gray.shape[0]
    rows = gray.shape[1]
    hole = np.ones(gray.shape,np.uint8)*0
    cnts = Max_area_cnt(contours)
    X=[]
    Y=[]
    #获取轮廓的坐标（x，y）
    for cnt in cnts:
        [[x,y]] = cnt
        X.append(x)
        Y.append(y)
    Y = np.array(Y)
    X = np.array(X)
    rr, cc=polygon(Y,X)
    draw.set_color(hole,[rr,cc],[255])#根据轮廓绘制蒙版
    crop = np.zeros((cols, rows), np.uint8)
    crop=cv.cvtColor(crop,cv.COLOR_GRAY2BGR)#使得能够与原始彩色图像转换
    for row in range(hole.shape[0]):
        for col in range(hole.shape[1]):
        # 如果掩图的像素不等于0，则dst(x,y) = scr(x,y)
            if hole[row, col] != 0:
                 # dst_image和scr_Image一定要高宽通道数都相同，否则会报错
                crop[row, col] = img_o[row, col]  
                leaf = img_o[row, col] 
                leafs.append(leaf)
            # 如果掩图的像素等于0，则dst(x,y) = 0
            elif hole[row, col] == 0:
                crop[row, col] = 0
    leafs = np.array([leafs])
    return leafs
def Color_feature(image):
    """颜色特征提取
    HSV颜色模型
    """
    img = Area_extract(image)
    color_features = []
    color_feature = []
    
    #HSV
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)
   
    # The first central moment - average 
    h_mean = int(np.mean(h)*2.0)  # np.sum(h)/float(N)
    s_mean = int(np.mean(s)*100/255.0)  # np.sum(s)/float(N)
    v_mean = int(np.mean(v)*100/255.0) # np.sum(v)/float(N)
    color_feature.extend([h_mean, s_mean, v_mean])
    # The second central moment - standard deviation
    h_std = round(np.std(h)*2.0,2)  # np.sqrt(np.mean(abs(h - h.mean())**2))
    s_std = round(np.std(s)*100/255,2)  # np.sqrt(np.mean(abs(s - s.mean())**2))
    v_std = round(np.std(v)*100/255,2)  # np.sqrt(np.mean(abs(v - v.mean())**2))
    color_feature.extend([h_std, s_std, v_std])
    # The third central moment - the third root of the skewness
    h_t = np.mean((h*2.0 - h.mean()*2.0)**3)
    s_t = np.mean((s*100/255.0 - s.mean()*100/255.0)**3)
    v_t = np.mean((v*100/255.0 - v_mean)**3)
    if h_t > 0:
        h_skewness = round(h_t**(1.0/3),2)
    else:
        h_t = -h_t
        h_skewness = round(h_t**(1.0/3),2)
        h_skewness = -h_skewness
    if s_t > 0:
        s_skewness = round(s_t**(1.0/3),2)
    else:
        s_t = -s_t
        s_skewness = round(s_t**(1.0/3),2)
        s_skewness = -s_skewness
    if v_t > 0:
        v_skewness = round(s_t**(1.0/3),2)
    else:
        v_t = -v_t
        v_skewness = round(v_t**(1.0/3),2)
        v_skewness = -v_skewness
    color_feature.extend([h_skewness, s_skewness, v_skewness])
    
    color_features.append(color_feature)
    return color_feature
def flatten(lst):
    """列表降维，减少方括号"""
    result = []
    def fly(lst):
        for item in lst:
            if isinstance(item, collections.Iterable) and not isinstance(item, (str, bytes)):
                fly(item)
            else:
                result.append(item)
                
    fly(lst)
    return result
def Bud_index(image, hmin, smin, vmin, hmax, smax,vmax):
    """对茶芽图片提取茶芽茸毛密度
    7.0更新
    """
    bud_image = Area_extract(image)
    bud=[]
    for a in bud_image[0,]:
        b,g,r = a 
        a = [[[b,g,r]]]
        a = np.uint8(a)
        hsv = cv.cvtColor(a, cv.COLOR_BGR2HSV)
        h, s, v = cv.split(hsv)
        # https://blog.csdn.net/taily_duan/article/details/51506776
        # white h:0-180, s:0-30, v:221-255
        # green h:35-77, s:43-255, v:46-255
        if hmin <= h <= hmax and smin <= s <= smax and vmin <= v <= vmax:
            bud.append(hsv)
    bud = len(bud)
    area = len(bud_image[0,])
    index = bud/area*100
    buds=[bud,area,index]
    return buds
def Bud_pixel(image):
    """对茶芽图片提取茶芽茸毛密度
    4.0更新
    """
    bud_image = Area_extract(image)
    bud_pixel = []
    hlist = []
    slist = []
    vlist = []
    for a in bud_image[0,]:
        b,g,r = a
        a = [[[b,g,r]]]
        a = np.uint8(a)
        hsv = cv.cvtColor(a, cv.COLOR_BGR2HSV)
        h, s, v = cv.split(hsv)
        hlist.append(h)
        slist.append(s)
        vlist.append(v)
#        bud_pixel.append(hsv)
    return(hlist,slist,vlist)
def Bud_triMask(image, hmin, smin, vmin, hmax, smax,vmax):
    # https://blog.csdn.net/qq_40784418/article/details/106347196
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    lower = np.array([hmin, smin, vmin])
    upper = np.array([hmax, smax, vmax])
    mask = cv.inRange(hsv,lower,upper)
    imgResult = cv.bitwise_and(image,image,mask=mask)
    #return mask
    return imgResult
def output_csv(name, data,headers = ['编号','周长','面积','长轴','短轴','长宽比','范围','坚实度','等效直径','圆形度',
               'contrast_0','dissimilarity_0','homogeneity_0','correlation_0','ASM_0','entropy_0',
                ' h_mean', 's_mean','v_mean', 'h_std', 's_std', 'v_std','h_skewness','s_skewness','v_skewness']):
    """输出CSV文件"""
    rows = data
    with open(name,'w',newline='')as f:
        f_csv = csv.writer(f)
        f_csv.writerow(headers)
        f_csv.writerows(rows)

def Coin_shape(image):
    """利用硬币参照物能对手机拍摄的叶片图像进行形状特征的提取
    4.0更新
    """
    gray,binary = Preprocessing(image)
    img_rule=cv.inRange(image,(130,130,130),(255,255,255))#利用色差提取硬币轮廓
    closing_std = cv.morphologyEx(img_rule, cv.MORPH_CLOSE, kernal)
    opening_std = cv.morphologyEx(closing_std, cv.MORPH_OPEN, kernal)
    contours_std, hierarchy = cv.findContours(opening_std,
                                          cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cnt_std = Max_area_cnt(contours_std)
    x,y,w,h = cv.boundingRect(cnt_std) 
    long_axis = round(h/118,2)
    k = long_axis/2.49
    img = np.zeros((h+100,w+100), np.uint8)#将硬币区域彻底变黑
    binary[y-50:(y+h)+50,x-50:(x+w)+50] = img
    contours, hierarchy = cv.findContours(binary,
                                              cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE) 
    cnt = Max_area_cnt(contours)
    #周长
    perimeter = cv.arcLength(cnt,True)
    perimeter_cm = round(perimeter/118/k,2)
     
    #面积
    area = cv.contourArea(cnt)
    area_cm = round(area/13924/k/k,2)
     
    #长轴短轴
    x,y,w,h = cv.boundingRect(cnt)
    long_axis = round(h/118/k,2)
    minor_axis = round(w/118/k,2)
     
    #长宽比
    aspect_ratio =round(float(w)/h,2) 
     
    #范围 范围是轮廓区域与边界矩形区域的比值。
    area = cv.contourArea(cnt)
    x,y,w,h = cv.boundingRect(cnt)
    rect_area = w*h
    extent = round(float(area)/rect_area,2)
     
     #坚实度 坚实度是目标面积与其凸包面积之比。
    hull = cv.convexHull(cnt)
    hull_area = cv.contourArea(hull)
    solidity = round(float(area)/hull_area,2)
     
     #等效直径 等效直径是面积与轮廓面积相同的圆的直径。
    area = cv.contourArea(cnt)
    equi_diameter = np.sqrt(4*area/np.pi)
    equi_diameter_cm = round(equi_diameter/118/k,2)
     
    #圆形度e为1时，图形即为圆形；e越小，图形越不规律，与圆形的差距越大。e=（4π*面积）/（周长*周长）
    e = 4*np.pi*area_cm/perimeter_cm/perimeter_cm
    e = round(1/e,2)
     
     
     
    list = [perimeter_cm,area_cm,long_axis,minor_axis,aspect_ratio,
      extent,solidity,equi_diameter_cm,e]
    return list
def Coin_color(image):
    """根据硬币作为参考提取手机照片中的叶片颜色特征
    4.0更新
    """
    gray,binary = Preprocessing(image)
    img_rule=cv.inRange(image,(130,130,130),(255,255,255))#利用色差提取硬币轮廓
    closing_std = cv.morphologyEx(img_rule, cv.MORPH_CLOSE, kernal)
    opening_std = cv.morphologyEx(closing_std, cv.MORPH_OPEN, kernal)
    img_leaf = cv.bitwise_xor(opening_std,binary)#用异位处理，保留叶片区域轮廓，去掉硬币区域轮廓。
    contours_std, hierarchy = cv.findContours(opening_std,
                                              cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cnt_std = Max_area_cnt(contours_std)
    x,y,w,h = cv.boundingRect(cnt_std)
    img = np.zeros((h+100,w+100), np.uint8)#将硬币区域彻底变黑
    img_leaf[y-50:(y+h)+50,x-50:(x+w)+50] = img
    contours, hierarchy = cv.findContours(img_leaf,
                                              cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE) 
    cnt = Max_area_cnt(contours)
    x,y,w,h = cv.boundingRect(cnt)
    img_leaf = cv.bitwise_and(image,image,mask = img_leaf)#提取叶片部分，并让其他区域全部为(0,0,0)
    img_leaf = img_leaf[y-50:(y+h)+50,x-50:(x+w)+50]
    img_leaf = cv.cvtColor(img_leaf,cv.COLOR_BGR2HSV)
    h,s,v = cv.split(img_leaf)
    
    h = np.resize(h,h.shape[1]*h.shape[0])#降维
    #将0元素全部去除,保留彩色部分
    h = pd.DataFrame(h).replace(0,np.NAN)
    h.dropna(inplace = True)
    h = np.array(h).astype(np.int32).reshape(1,-1)[0]
    
    s = np.resize(s,s.shape[1]*s.shape[0])#降维
    #将0元素全部去除,保留彩色部分
    s = pd.DataFrame(s).replace(0,np.NAN)
    s.dropna(inplace = True)
    s = np.array(s).astype(np.int32).reshape(1,-1)[0]
    
    v = np.resize(v,v.shape[1]*v.shape[0])#降维
    #将0元素全部去除,保留彩色部分
    v1 = pd.DataFrame(v).replace(0,np.NAN)
    v1.dropna(inplace = True)
    v2 = np.array(v1).astype(np.int32).reshape(1,-1)[0]
    
    color_feature = []
    h_mean = int(np.mean(h)*2.0)  # np.sum(h)/float(N)
    s_mean = int(np.mean(s)*100/255.0)  # np.sum(s)/float(N)
    v_mean = int(np.mean(v2)*100/255.0) # np.sum(v)/float(N)
    color_feature.extend([h_mean, s_mean, v_mean])
    # The second central moment - standard deviation
    h_std = round(np.std(h)*2.0,2)  # np.sqrt(np.mean(abs(h - h.mean())**2))
    s_std = round(np.std(s)*100/255,2)  # np.sqrt(np.mean(abs(s - s.mean())**2))
    v_std = round(np.std(v2)*100/255,2)  # np.sqrt(np.mean(abs(v - v.mean())**2))
    color_feature.extend([h_std, s_std, v_std])
    # The third central moment - the third root of the skewness
    h_t = np.mean((h*2.0 - h.mean()*2.0)**3)
    s_t = np.mean((s*100/255.0 - s.mean()*100/255.0)**3)
    v_t = np.mean((v2*100/255.0 - v_mean)**3)
    if h_t > 0:
        h_skewness = round(h_t**(1.0/3),2)
    else:
        h_t = -h_t
        h_skewness = round(h_t**(1.0/3),2)
        h_skewness = -h_skewness
    if s_t > 0:
        s_skewness = round(s_t**(1.0/3),2)
    else:
        s_t = -s_t
        s_skewness = round(s_t**(1.0/3),2)
        s_skewness = -s_skewness
    if v_t > 0:
        v_skewness = round(s_t**(1.0/3),2)
    else:
        v_t = -v_t
        v_skewness = round(v_t**(1.0/3),2)
        v_skewness = -v_skewness
    color_feature.extend([h_skewness, s_skewness, v_skewness])
    return color_feature
def Coin_texture(image):
    gray,binary = Preprocessing(image)
    img_rule=cv.inRange(image,(130,130,130),(255,255,255))#利用色差提取硬币轮廓
    closing_std = cv.morphologyEx(img_rule, cv.MORPH_CLOSE, kernal)
    opening_std = cv.morphologyEx(closing_std, cv.MORPH_OPEN, kernal)
    img_leaf = cv.bitwise_xor(opening_std,binary)#用异位处理，保留叶片区域轮廓，去掉硬币区域轮廓。
    #计算标准物的轮廓
    contours_std, hierarchy = cv.findContours(img_rule,
                                              cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)  
    areas_std = []
    for cnt_std in contours_std:
        A_std = cv.contourArea(cnt_std)
        areas_std.append(A_std)
    max_id = areas_std.index(max(areas_std))#选出轮廓面积最大的轮廓
    cnt_std = contours_std[max_id]
    x,y,w,h = cv.boundingRect(cnt_std)
    img = np.zeros((h+100,w+100), np.uint8)#将硬币区域彻底变黑
    img_leaf[y-50:(y+h)+50,x-50:(x+w)+50] = img
    contours, hierarchy = cv.findContours(img_leaf,
                                              cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE) 
    areas = []
    for cnt in contours:
        A = cv.contourArea(cnt)
        areas.append(A)
    max_id = areas.index(max(areas))
    cnt = contours[max_id]
    x,y,w,h = cv.boundingRect(cnt)
    img_leaf = gray[y-50:(y+h)+50,x-50:(x+w)+50]
    
    texture = Texture_feature(img_leaf)
    return texture
def Coin_bud(image):
    gray,binary = Preprocessing(image)
    img_rule=cv.inRange(image,(130,130,130),(255,255,255))#利用色差提取硬币轮廓
    closing_std = cv.morphologyEx(img_rule, cv.MORPH_CLOSE, kernal)
    opening_std = cv.morphologyEx(closing_std, cv.MORPH_OPEN, kernal)
    img_leaf = cv.bitwise_xor(opening_std,binary)#用异位处理，保留叶片区域轮廓，去掉硬币区域轮廓。
    #计算标准物的轮廓
    contours_std, hierarchy = cv.findContours(img_rule,
                                              cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)  
    areas_std = []
    for cnt_std in contours_std:
        A_std = cv.contourArea(cnt_std)
        areas_std.append(A_std)
    max_id = areas_std.index(max(areas_std))#选出轮廓面积最大的轮廓
    cnt_std = contours_std[max_id]
    x,y,w,h = cv.boundingRect(cnt_std)
    img = np.zeros((h+100,w+100), np.uint8)#将硬币区域彻底变黑
    img_leaf[y-50:(y+h)+50,x-50:(x+w)+50] = img
    contours, hierarchy = cv.findContours(img_leaf,
                                              cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE) 
    areas = []
    for cnt in contours:
        A = cv.contourArea(cnt)
        areas.append(A)
    max_id = areas.index(max(areas))
    cnt = contours[max_id]
    x,y,w,h = cv.boundingRect(cnt)
    img_leaf = image[y-50:(y+h)+50,x-50:(x+w)+50]
    
    buds = Bud_index(img_leaf)

    return buds
