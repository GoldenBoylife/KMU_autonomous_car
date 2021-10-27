#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2, random, math, copy, time
import rospy, rospkg
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from xycar_msgs.msg import xycar_motor
from std_msgs.msg import Int8
from collections import deque
import calibrate_image
#from XycarSensor import XycarSensor
#from ARController import ARController


class PID():

    def __init__(self, kp, ki, kd):
        self.Kp=kp
        self.Ki=ki
        self.Kd=kd
        self.p_error=0.0
        self.i_error=0.0
        self.d_error=0.0

    def pid_control(self, cte):
        self.d_error=cte-self.p_error
        # d 값은 현재 CTE 값과 이전 p값의 차이를 이용
        self.p_error=cte
        # p 값은 CTE 값을 그대로 적용
        self.i_error+=cte
        # i 값은 CTE 값을 계속 더해서 누적한 값을 적용

        return self.Kp*self.p_error+self.Ki*self.i_error+self.Kd*self.d_error

class SlidingWindow():
    def __init__(self):
        # self.xycar_image = np.empty(shape=[0])
        self.Width = 640
        self.Height = 480
        # self.bridge = CvBridge()

        self.left_line_que = deque([0, 0, 0, 0, 0])
        self.right_line_que = deque([320, 320, 320, 320, 320])

        self.state = "curve"

        self.warp_img_w = 320 # 가로 이미지 크기(열)
        self.warp_img_h =240 # 세로 이미지 크기(행)
        self.warpx_margin =18 # x 마진 
        self.warpy_margin =3 # y 마진
        self.nwindows =9 #슬라이딩 윈도우 개수
        self.margin =12  #슬라이딩 윈도우 넓이 기본 12
        self.minpix =5   #선을 그리기 위해 최소한 있어야 할 점의 개수  기존: 5

        # 원본 이미지에서 사다리꼴 좌표 1,2,3,4번 점(좌상,좌하,우상,우하)
        self.leftup = 105, 320
        self.leftdown = 10, 371
        self.rightup = 445, 323 # 457, 300
        self.rightdown = 529, 374 # 545, 355
        # 사다리꼴 좌표
        self.warp_src = np.array([
                    [self.leftup], 
                    [self.leftdown],
                    [self.rightup],
                    [self.rightdown],
                    ], dtype=np.float32)
        
        ##320x240 이미지로 직사각형 이미지로 만듬. 1,2,3,4번 점
        # 와핑된 이미지 좌표
        self.warp_dist = np.array([
                    [0,0],
                    [0,self.warp_img_h],
                    [self.warp_img_w,0],
                    [self.warp_img_w, self.warp_img_h],
                    ], dtype=np.float32)
        
        # camera calibration
        self.mtx = np.array([
            [357.1634, 0.0,  317.44566], 
            [0.0, 357.69039, 237.29034], 
            [0.0, 0.0, 1.0]
        ])

        self.dist = np.array([-0.288943, 0.053825, 0.000197, -0.006636, 0.0])
        self.cal_mtx, self.cal_roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (self.Width, self.Height), 1, (self.Width, self.Height))

        # rospy.init_node('drive')
        
        # 이미지 받기
        # self.image_sub = rospy.Subscriber("/usb_cam/image_raw", Image, self.img_callback)

    # def img_callback(self, data):
    #     self.xycar_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

    def calibrate_image(self, frame):
        # start_time = time.time()
        tf_image = cv2.undistort(frame, self.mtx, self.dist, None, self.cal_mtx)
        
        x, y, w, h = self.cal_roi
        # cv2.imshow("tf_image", tf_image)
        # vertices = np.array([[(x, y+h), (x, y), (x+w, y), (x+w, y+h)]], dtype=np.int32)
        #tf_image = self.ROI(tf_image, [vertices])
        tf_image = tf_image[y:y+h, x:x+w]
        # cv2.imshow("tf_image2", tf_image)
        # end_time = time.time()
        # print("process_time : ", end_time - start_time)

        return cv2.resize(tf_image, (self.Width, self.Height), interpolation=cv2.INTER_LINEAR)

    def warp_image(self, img):
        size = (self.warp_img_w, self.warp_img_h)
        M = cv2.getPerspectiveTransform(self.warp_src, self.warp_dist) # 사다리꼴에서 직사각형
        Minv = cv2.getPerspectiveTransform(self.warp_dist, self.warp_src) # 직사각형에서 사다리꼴
        warp_img = cv2.warpPerspective(img, M, size, flags=cv2.INTER_LINEAR)

        return warp_img, M, Minv

    def warp_process_image(self, img):
        #------------------------------sliding_window------------------------------#
        
        blur = cv2.GaussianBlur(img,(5,5),0)
        L, _, _ = cv2.split(cv2.cvtColor(blur, cv2.COLOR_BGR2LAB))
        #_, L , _ = cv2.split(cv2.cvtColor(blur, cv2.COLOR_BGR2HLS))
        
        # canny edge
        low_threshold = 40
        high_threshold = 70
        edge_img = cv2.Canny(np.uint8(L), low_threshold, high_threshold)

        # _,lane = cv2.threshold(L, 160, 255, cv2.THRESH_BINARY)
        # cv2.imshow("L", L)
        kernel = np.ones((1,1),np.uint8)
        lane = cv2.dilate(edge_img, kernel) # 팽창 즉 이진화한 것의 흰색부분을 더욱 크게 만들어주는 역할
        cv2.imshow("binary", lane)

        histogram = np.sum(lane[215:,:], axis=0)
        midpoint = np.int(histogram.shape[0]/2)

        center = 40
        left_point = midpoint - center
        right_point = midpoint + center
        
        # 첫번째 슬라이딩윈도우의 왼쪽차선, 오른쪽 차선의 x좌표
        leftx_current = np.argmax(histogram[:left_point])
        rightx_current = np.argmax(histogram[right_point:]) + right_point
        print("real:",rightx_current)

        # Queue
        if len(self.left_line_que) >= 5:
            if self.state == "curve":
                if not self.left_line_que[-1] - 40 <= leftx_current <= self.left_line_que[-1] + 40:
                    leftx_current = self.left_line_que[-1]
                if not self.right_line_que[-1] - 40 <= rightx_current <= self.right_line_que[-1] + 40:
                    rightx_current = self.right_line_que[-1]
            else:
                if not self.left_line_que[-1] - 20 <= leftx_current <= self.left_line_que[-1] + 32:
                    leftx_current = self.left_line_que[-1]
                if not self.right_line_que[-1] - 32 <= rightx_current <= self.right_line_que[-1] + 25:
                    rightx_current = self.right_line_que[-1]
            self.left_line_que.popleft()
            self.right_line_que.popleft()
        
        if rightx_current <= 285:
            leftx_current = 0
        if leftx_current >= 40:
            rightx_current = 320

        self.left_line_que.append(leftx_current)
        self.right_line_que.append(rightx_current)

        # if rightx_current <= 284 or leftx_current >= 45: # 288, 35
        #     self.state = "curve"
        # else:
        #     self.state = "st"
        print("r:",rightx_current)
        print("l",leftx_current)
        print("state:", self.state)
                
        # if np.argmax(histogram[midpoint:]) == 0:
        #         rightx_current = np.argmax(histogram[:midpoint])+230
        # else:
        #     rightx_current = np.argmax(histogram[midpoint:]) + midpoint

        #오른쪽 절반 구역에서 흰색 픽셀의 개수가 가장 많은 위치를 슬라이딩 윈도우의 오른쪽 시작 위치로잡기.


        ##차선있는 곳에 박스를 9개 쌓는다.
        window_height = np.int(lane.shape[0]/self.nwindows)
        nz= lane.nonzero()
        #numpy 요소 중에서 0이 아닌 numpy의 위치만 인덱스를 찾아서 nz로 저장함.
        left_lane_inds = []
        right_lane_inds =[]
        lx, ly, rx, ry = [], [], [], []
        out_img = np.dstack((lane,lane,lane))*255
        #np.dstack((lane,lane,lane)) 자체는 binary와 같은 창이 뜬다.
        #out_img는 (240,320) 크기의 완전 검은 창 하나가 뜬다.  아마 255를 다 곱해서 검은색 뜨는듯.
        
        left_wrong_point_cnt = 0
        right_wrong_point_cnt = 0

        ##9개를 사각형을 그리는 루프
        for window in range(self.nwindows):      #9번 for문 돈다.
            ##위아래 높이 계산
            win_yl = lane.shape[0] - (window+1)* window_height
            #window에는 0~8까지 순서대로 들어갈 것이다. left의 y높이에 따라서 9개 점이군.
            win_yh = lane.shape[0] - window*window_height
            ##  왼쪽차선의 가운데 현재값을 구하고 그다음 꺼 구해서 계속 반복함.
            ##margin은 12로 녹색박스 크기 뜻함.
            ##      왼쪽 차선쪽 윈도우
            win_xll = leftx_current - self.margin
            win_xlh = leftx_current + self.margin
            ##      오른쪽 차선쪽 윈도우
            win_xrl = rightx_current - self.margin
            win_xrh = rightx_current + self.margin
            ##녹색박스 그린다.
            cv2.rectangle(out_img,(win_xll, win_yl), (win_xlh,win_yh), (0,255,0),2)
            #rectangle(영상,왼상단점 위치, 우하단점 위치, 색, 두께)
            cv2.rectangle(out_img,(win_xrl, win_yl), (win_xrh,win_yh), (0,255,0),2)

            ##녹색 박스 하나 안에 있는 흰색 픽셀의 x좌표를 모두 모은다. 쌍곡선 그리려고 하는듯
            ##  왼쪽선
            good_left_inds = ((nz[0] >= win_yl ) & (nz[0] < win_yh) & (nz[1] >=win_xll)
                        & (nz[1] < win_xlh)).nonzero()[0]
                        #nz[0]는 numpy중 0이 아닌 것만 있는 인덱스중 x값들 쪽만 들어가 있다.
            ##  오른쪽선
            good_right_inds = ((nz[0] >= win_yl) & (nz[0] < win_yh) &
                            (nz[1] >= win_xrl) & (nz[1] < win_xrh)).nonzero()[0]

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            ##x좌표 리스트에서 흰색점이 5개 이상인 경우 x좌표의 평균값을 구한다.
            ##평균값 가지고 그 다음 박스 위치가 정해진다.
            ##minpix는 5이고 5개 이상
            if len(good_left_inds) > self.minpix:
                leftx_current = np.int(np.mean(nz[1][good_left_inds]))

            if len(good_right_inds) >self.minpix:
                rightx_current = np.int(np.mean(nz[1][good_right_inds]))

            if self.left_line_que[-1] == leftx_current:
                left_wrong_point_cnt += 1
            if self.right_line_que[-1] == rightx_current:
                right_wrong_point_cnt += 1
            ##슬라이딩 윈도우의 중심점(x좌표)를 담아둔다. 결국 9개 모두 모은다.
            lx.append(leftx_current)            #왼쪽 차선 중심점 모으고  나중에 2차함수에 씀.
            ly.append((win_yl+ win_yh)/2)
            rx.append(rightx_current)
            ry.append((win_yl + win_yh)/2)
        #슬라이딩 윈도우의 중심점 x좌표들이 모여있을테니까  그것들의 합을 나눠서 평균길이 구한다.
        avg_lx = sum(lx)/self.nwindows
        avg_rx = sum(rx)/self.nwindows
        
        # if left_wrong_point_cnt >= 8:
        #     self.left_line_que[-1] = 0
        
        # if right_wrong_point_cnt >= 8:
        #     self.right_line_que[-1] = 320

        print("last:", lx[-1], rx[-1])
        if rx[-1] <= 269 or lx[-1] >= 48:
            self.state = "curve"
        else:
            self.state = "st"

        if len(set(lx)) <= 1 and self.left_line_que[-1] != 0:
            self.left_line_que[-1] = 15
        if len(set(rx)) <= 1 and self.right_line_que[-1] != 320: 
            self.right_line_que[-1] = 305 

        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
        ##슬라이딩 윈도우의 중심점(x좌표) 9개를 가지고 2차 함수를 만들어 낸다.
        lfit = np.polyfit(np.array(ly),np.array(lx),2)
        # left쪽 중심점 좌표 9개를 가지고 그 9개 차선을 돌아가는 그래프를 하나 만든다.
        #polyfit(x좌표, y좌표, 2차수)로 2차식으로 표현하기 위해서 주어진 데이터의 최소 제곱 구한다.
        #애러의 제곱의 합을 최소화하는 공업수학적 방법이라고 한다. "최소제곱법"
        rfit = np.polyfit(np.array(ry),np.array(rx),2)
        ##디버깅 작업 기존 하얀색 차선을 각각 다른 색으로 변경한다.
        out_img[nz[0][left_lane_inds], nz[1][left_lane_inds]] = [255,0,0]
        #   왼쪽 차선 쪽에서 인식된 흰색이 전부 파란색으로 바뀐다.
        out_img[nz[0][right_lane_inds], nz[1][right_lane_inds]] = [0,0,255]
        #   오른 차선 쪽에서 인식된 흰색이 전부 빨간색으로 바뀐다.

        cv2.imshow("viewer",out_img)
        #검은 배경에 녹색 박스 9개와 선 2개가 그려진다.

        return lfit, rfit, avg_lx, avg_rx #, sliding_mid

    def weighted_moving_average(self, n, pos_deque):
        weight, avg = 0, 0
        for i in range(1, n+1):
            weight += i*pos_deque[i-1]
            avg += i
        pos_avg = weight/avg
        return pos_avg

    def draw_lane(self, image, warp_img, Minv, left_fit, right_fit):
        yMax= warp_img.shape[0]
        ploty = np.linspace(0, yMax -1, yMax)
        color_warp = np.zeros_like(warp_img).astype(np.uint8)
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        ##이차함수  x= ay2+by+c를 이용해서 사다리꼴 이미지 외곽선 픽셀 좌표 계산하기
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right =np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        ##사다리꼴 이미지를 칼라로 그리고 역원근변환해서 원본이미지와 오버레이 한다.
        ##역원근 변환 : 직사각형 -> 사다리꼴
        color_warp = cv2.fillPoly(color_warp, np.int_([pts]), (0,255,0))
        newwarp = cv2.warpPerspective(color_warp, Minv, (self.Width, self.Height))
        # image =cv2.circle(image,(leftup),5,(255,0,0),5)
        # image =cv2.circle(image,(leftdown),5,(255,255,0),5)
        # image =cv2.circle(image,(rightup),5,(255,0,255),5)
        # image =cv2.circle(image,(rightdown),5,(0,255,0),5)

        return cv2.addWeighted(image,1,newwarp, 0.3,0)

    def start(self, xycar_image):
        White_detect = False
        while not xycar_image.size == (self.Width*self.Height*3):
            print("not defined image")
            continue
        # start_time = time.time()
        # image = xycar_image ## 실제 차에서 사용
        
        #roi
        
        vertices = np.array([[(10, 430), (20, 300), (620, 300), (630, 430)]], dtype=np.int32)
        roi_frame = self.ROI(xycar_image, [vertices])
        cv2.imshow("roi", roi_frame)
        # cv2.imshow("ori", xycar_image)
        
        # calibration
        start_time = time.time()
        image = calibrate_image.calibrate_image(roi_frame, self.mtx, self.dist, self.Width, self.Height) ## bag파일용
        end_time = time.time()
        print("process_time : ", end_time - start_time)
        
        # wapping
        warp_img, M, Minv = self.warp_image(image)
        cv2.imshow("wapping", warp_img)

        # stop line check
        stop = stopline_detect(warp_img)
        print("stop",stop)

        # image processing 
        left_fit, right_fit, avg_lx, avg_rx = self.warp_process_image(warp_img)
        
        # green lane draw
        lane_img = self.draw_lane(image, warp_img, Minv, left_fit, right_fit)
        
        # steering control
        center = 160
        error = center - (avg_lx + avg_rx)/2
        print("err:", error)
        # pid = PID(0.28, 0.001, 0.08)
        # angle = pid.pid_control(error)
        

        # if abs(error) <= 15 and self.state == "st":
        #         pid = PID(0.01, 0.001, 0.02)
        #         angle = pid.pid_control(error)
        #         steer_angle = (-angle) -1
        #         speed = 20
        # else:
        if abs(error) < 20:
            pid = PID(0.015, 0.01, 0.002)
            angle = pid.pid_control(error)
            steer_angle = ((-angle)-1.0)
            speed = 10
        else:
            if self.state == "curve":
                pid = PID(1.3, 0, 0.08)
                angle = pid.pid_control(error)
                steer_angle = ((-angle)-1.0)
                speed = 10
            else:
                pid = PID(0.12, 0.001, 0.02)
                angle = pid.pid_control(error)
                steer_angle = ((-angle)-1.0)*0.8
                speed = 10

        cv2.imshow("sliding", lane_img)

        return steer_angle, speed, stop
    
    def ROI(self, frame, vertices):
        # blank mask:
        mask = np.zeros_like(frame)
        # fill the mask
        cv2.fillPoly(mask, vertices, (255,255,255))

        # now only show the area that is the mask
        masked = cv2.bitwise_and(frame, mask)
        return masked

##-----------------정지선 관련 함수 ----start-----------##
def stopline_detect(cal_image):
    ##roi-> gray->thr -> contour ->폭으로 
    #cv2.imshow("temp",stopline_roi)
    gray= cv2.cvtColor(cal_image,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    _, thr = cv2.threshold(blur, 150,255,cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    erod = cv2.erode(thr, k)
    
    _,contours, _ =cv2.findContours(erod,cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.imshow('erod',erod)
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

            if cont_xwidth > 200 and cont_ywidth > 50:             
                cal_image = setLabel(cal_image, cont, 'stopline')
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
# if __name__ == '__main__':
    
#     avg_lx = 0
#     avg_rx = 0
#     angle = 0
#     motor_control = xycar_motor()
#     line_cte = Int8()
#     pub = rospy.Publisher("xycar_motor", xycar_motor, queue_size = 1)
#     sliding = SlidingWindow()
#     rate = rospy.Rate(30)
#     print("start")
#     while not rospy.is_shutdown():
        
#         start_time = time.time()
#         angle, speed = sliding.start()
#         end_time = time.time()
#         # if end_time - start_time >= 0.03:
#         #     print("process_time : ", end_time - start_time) ## 0.0333안에 처리가 되는지 확인필요함
#         # motor_pub(angle, speed)

#         cv2.waitKey(1)
#         rate.sleep()
