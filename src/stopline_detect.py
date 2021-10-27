#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2, random, math, copy, time
import rospy, rospkg
from cv_bridge import CvBridge

##-----------------정지선 관련 함수 ----start-----------##
def stopline_detect(cal_image, low_threshold_value):
    ##roi-> gray->thr -> contour ->폭으로 
    #cv2.imshow("temp",stopline_roi)
    gray= cv2.cvtColor(cal_image,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    _, thr = cv2.threshold(blur, 150,255,cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    erod = cv2.erode(thr, k)
    
    _,contours, _ =cv2.findContours(erod,cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.imshow('erod',erod)
    for cont in contours:
        approx = cv2.approxPolyDP(cont, cv2.arcLength(cont,True) * 0.02, True)
        vtc = len(approx)
        cont_xmin= 300
        cont_xmax =0
        cont_xwidth = 0
        cont_ymin = 300
        cont_ymax = 0
        cont_ywidth = 0
        if vtc ==4:
            for j in cont:
                i = j[0][0]
                k = j[0][1]                    
                                    
                if cont_xmax < i:
                    cont_xmax = i
                if cont_xmin > i:
                    cont_xmin = i
                if cont_ymax < k:
                    cont_ymax = k
                if cont_ymin > k:
                    cont_ymin = k    
                                                                

            cont_xwidth = cont_xmax -cont_xmin
            cont_ywidth = cont_ymax - cont_ymin

            #cv2.imshow("image",cal_image)
            print()                
            print("x, y width", cont_xwidth,cont_ywidth)                                
            print()
            if cont_xwidth > 200 and cont_ywidth > 50:
                print("x, y width", cont_xwidth,cont_ywidth)                
                cal_image = setLabel(cal_image, cont, 'SSSstopline')
                cv2.imshow("image",cal_image)
                return True
    return False
def setLabel(img, pts, label):
        """
        도형찾고 라벨링
        """

        (x, y, w, h) = cv2.boundingRect(pts)
        pt1 = (x, y)
        pt2 = (x+w, y+h)
        cv2.rectangle(img, pt1, pt2, (0,255,0), 2)
        cv2.putText(img, label, (pt1[0],pt1[1]-3), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255))
        return img

##-----------------정지선 관련 함수 ----end-----------##

