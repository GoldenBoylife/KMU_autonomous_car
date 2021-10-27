#! /usr/bin/env python
# -*- coding:utf-8 -*-

import cv2, time
import rospy
import numpy as np
from xycar_msgs.msg import xycar_motor
from XycarSensor import XycarSensor
from sliding_window1 import SlidingWindow
#from StopLineDetector import StopLineDetector
#from ObstacleDetector import ObstacleDetector
#from ModeController import ModeController
from ARController import ARController
from sliding_window1 import PID
from stopline_detect import stopline_detect

def motor_pub(angle, speed):
    global pub
    global motor_control

    motor_control.angle = angle
    motor_control.speed = speed

    pub.publish(motor_control)

def findparking(ranges):
    global mode
    ranges = ranges[320:360]
    ranges = np.array(ranges)
    
    print(ranges)
    if np.count_nonzero((ranges > 0.0 ) & (ranges < 0.7)) > 30 :
        print('start parking...')
        mode = "parallelpark"

def parallelpark():
    global mode
    for _ in xrange(33):
        motor_pub(7, 5)
        rate.sleep()
    for _ in xrange(40):
        motor_pub(50, -25)
        rate.sleep()
    for _ in xrange(32):
        motor_pub(-50, -25)
        rate.sleep()
    mode = "arparking"

def arparking():
    global mode
    angle, speed = ar_controller(sensor.ar_x, sensor.ar_y, sensor.ar_yaw)
    motor_pub(0, 0)
    rate.sleep()


if __name__ == '__main__':
    
    angle = 0
    motor_control = xycar_motor()
    
    rospy.init_node('drive')
    pub = rospy.Publisher("xycar_motor", xycar_motor, queue_size = 1)
    
    sliding = SlidingWindow()
    sensor = XycarSensor()
    ar_controller = ARController()
    rate = rospy.Rate(30)
    sensor.init(rate)
    print("start")
    mode = "lane_detect"
    while not rospy.is_shutdown():

        # if end_time - start_time >= 0.03:
        #     print("process_time : ", end_time - start_time) ## 0.0333안에 처리가 되는지 확인필요함

        if mode == "lane_detect":
            st_time = time.time()
            angle, speed, stop = sliding.start(sensor.image)
            motor_pub(angle, 0)
            if stop:
                mode = "stop"
            en_time = time.time()
            print(en_time-st_time)
            rate.sleep()
        elif mode == "stop":
            print("stop for 5s")
            for _ in xrange(12):

                motor_pub(0, 0)
                rate.sleep()
            mode = "lane_detect"
            print("go")
        elif mode == "findparking":
            findparking(sensor.lidar)
            angle, _, _ = sliding.start(sensor.image)
            motor_pub(angle, 6)
            rate.sleep()
        elif mode == "parallelpark":
            parallelpark()
        elif mode == "arparking":
            arparking()

        cv2.waitKey(1)