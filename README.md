# KMU_autonomous_car 
국민대와 자이트론이 주최한 자율주행 경진대회에서 예선 top15 통과, 본선 진출

<br>

## 1. Audi Team , charge
| name | in charge |
| ------------ | :-------------------------------------------------- |
| Hwang Hwidong (team leader) | Project Manager,Line detection (sliding_window), computer vision(stopline), track making |
| Kim jinwoo | Avoid obstacles, lidar sensor control, code refectoring |
| Oh gayoung | hough transformation, ROS, rosbag, picture,  |
| heo seock | Line detection (sliding_window), ARtag parking, code refectoring|
<br>

## 2. 프로젝트 목표
- 리눅스 환경에서 ROS로 자율주행 자동차 컨트롤하기.
- 모터제어, 센서제어, 영상처리(openCV), 사용함
- 학습한 내용을 적용하여 경진 대회 참가.

<br>

## Requiration
- xytron xycar a2모델( jetson tx2보드)
- 


## 사용된 프레임 워크 및 라이브러리

<br>
![python](https://img.shields.io/badge/python-2.7.0-orange?logo=python)
![ubuntu](https://img.shields.io/badge/ubuntu-18.04-violet?logo=ubuntu)
![openCV](https://img.shields.io/badge/opencv-skyblue?logo=opencv)

## 3. 맵 소개

<br>
![image](https://user-images.githubusercontent.com/81784631/138587480-fb860c24-c77f-4012-a9e8-c90fd122a90e.png)



## 4. 프로젝트 상세 내용. 

<br>
1. 차선인식 주행하기( sliding window) 사용, bird eye view)
![image](https://user-images.githubusercontent.com/81784631/138587326-37c62eb7-0f2e-42ce-8358-61098fd7a572.png)

2. 횡단보도 정지선 인식
![image](https://user-images.githubusercontent.com/81784631/138587420-fd79ea62-4697-43a8-af6b-8d4f9ccb785f.png)
![image](https://user-images.githubusercontent.com/81784631/138587156-71ce879c-71ff-48b0-b2d9-f0217727f8eb.png)

<br>
3. ARtag주차 
![image](https://user-images.githubusercontent.com/81784631/138587457-6ccf8997-ff3d-46d4-a2b3-4c21a660bcb7.png)
![image](https://user-images.githubusercontent.com/81784631/134535072-a88fe83d-f72e-4c02-aa27-ed2b10574f19.png)

4. 장애물 피하기
5.코드 리펙토링
![image](https://user-images.githubusercontent.com/81784631/134535195-c1b6e0bd-b13e-4d1d-a3a7-dae9a95a02a9.png)

## 5. 동영상 및 사진 자료.


