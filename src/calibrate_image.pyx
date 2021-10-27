#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2, random, math, copy, time
import rospy, rospkg
from cv_bridge import CvBridge


def calibrate_image(frame, mtx, dist, int Width, int Height):

    cdef int x, y, w, h
    
    cal_mtx, cal_roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (Width, Height), 1, (Width, Height))
    tf_image = cv2.undistort(frame, mtx, dist, None, cal_mtx)
    
    x, y, w, h = cal_roi

    tf_image = tf_image[y:y+h, x:x+w]


    return cv2.resize(tf_image, (Width, Height), interpolation=cv2.INTER_LINEAR)