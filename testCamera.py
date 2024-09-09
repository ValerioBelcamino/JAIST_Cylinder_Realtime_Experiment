#!/usr/bin/python3

import cv2
import threading
import os
from datetime import datetime
import time
from queue import Queue
import signal
import numpy as np
import inspect
import torch
from torchvision import transforms
from models import CHARberoViVit
from utils import CutBlackContour
import torch.nn.functional as F
from data_glove_driver import ImuDriver
import rospy
from std_msgs.msg import Bool

for i in range(20):
    vid1 = cv2.VideoCapture(i)
    if vid1.isOpened():
        print(f'{i}: {vid1.isOpened()}')
exit()


vid1 = cv2.VideoCapture(0)
fps = 29.0
interval = 1.0 / fps

transform1 = transforms.Compose([  
            CutBlackContour(left_margin=100, right_margin=65, top_margin=20, bottom_margin=0),
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=1),    
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.07427], std=[0.09232]),
        ])
transform2 = transforms.Compose([  
            CutBlackContour(left_margin=80, right_margin=80, top_margin=0, bottom_margin=40),
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=1),    
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.081102], std=[0.09952]),
            ])
    
while vid1.isOpened():
    videoOn1, videoOn2 = True, True
    print(f'vid1: {vid1.isOpened()}')

    start_time = time.time()

    if  vid1:
        ret1, frame1 = vid1.read()
    else:
        continue

    if ret1:
        #frame_buff_0 = frame1
        cv2.imshow(f'frame_{1}', frame1)
    else:
        print('Possible problem in camera1 driver')
        continue


    #vid1 = cv2.VideoCapture(2)
    #print(f'{vid1}: {vid1.isOpened()}')
    #vid2 = cv2.VideoCapture(0)
    #print(f'{vid2}: {vid2.isOpened()}')

    #while True:
    #    if  vid1:
    #        ret1, frame1 = vid1.read()

    #    if  vid2:
    #        ret2, frame2 = vid2.read()

    #    print(ret1)
    #    print(ret2)
    #    if ret1:
    #        frame_buff_0 = frame1
    #        cv2.imshow(f'frame_{1}', frame_buff_0)
    #    if ret2:
    #        frame_buff_1 = frame2
    #        cv2.imshow(f'frame_{2}', frame_buff_1)

    #    if cv2.waitKey(1) & 0xFF == ord('q'):
    #        break

    #input('aa')
    #exit()


   
