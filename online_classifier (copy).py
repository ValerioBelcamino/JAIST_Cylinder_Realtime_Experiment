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


# Function to update action timing
def update_action_timing(predicted_action):
    global current_action, action_start_time
 
    current_time = time.time()
 
    # If the action has changed or this is the first frame
    if predicted_action != current_action:
        if current_action is not None:
            # Calculate the duration of the previous action
            action_durations[current_action] = current_time - action_start_time
 
        # Start timing the new action
        current_action = predicted_action
        action_start_time = current_time
    else:
        # Update the duration for the current action
        action_durations[current_action] = current_time - action_start_time
 
    # Check if the current action duration exceeds the threshold
    if action_durations.get(current_action, 0) > action_threshold:
        return True  # Action has been performed for more than 0.5 seconds
    else:
        return False  # Action has not yet reached the threshold


def update_action_timing_window(predicted_action):
    global current_action, action_start_time, buffer
 
    current_time = time.time()

    
    buffer = torch.roll(buffer, shifts=-1, dims=0)
 
    buffer[-1] = predicted_action

    count = 0
    for b in buffer:
        if b == predicted_action:
            count = count +1

    #print(buffer)
    if count >= 10:
        buffer[:] = torch.nan
        return True
    else:
        return False



def signal_handler(sig, frame):
    ''' Handle the SIGINT signal (Ctrl+C) by stopping the threads and exiting the program. '''
    global thread_killer, driver_threads
    if not thread_killer:
        print('Ctrl+C detected. Stopping threads...')
        thread_killer = True
        

def deserialize_imu_data(data_str):
    data_str = data_str.split('\n')[0].split(',')[6:]

    data_np = torch.zeros((9))

    assert len(data_str) == 9

    for i, x in enumerate(data_str):
        data_np[i] = float(x)
    #print(data_np)
    #print(data_np.shape)
    return data_np
    
def IMUs_driver():
    ''' Start the IMU driver to listen for IMU data. '''
    global thread_killer, imu_buff_dict, hand
    imu_driver = ImuDriver(hand)

    while True:
        imu_data, not_socket_timeout = imu_driver.listener(thread_killer)
            
        #print(imu_data)
        sensor_id = imu_data.split(',')[1]
        #print(sensor_id)

        imu_data = deserialize_imu_data(imu_data)

        imu_buff_dict[sensor_id] = imu_data
        #print(imu_buff_dict)

        if thread_killer or not not_socket_timeout:
            for sensor_name in imu_sensor_names:
                imu_buff_dict[sensor_name] = None
            break

    print(f'Stopping thread {inspect.stack()[0][3]}...')



def IMUs_downsampler():
    ''' Start the IMU driver to listen for IMU data. '''
    global thread_killer, imu_buff_dict, imu_buff_downsampled, fps

    interval = 1.0 / fps

    means = np.array([-6.96805542e+02, -2.25853418e+03, -1.34595679e+03, -1.11699450e+00,
                    5.86712211e-02, -1.07630253e+00, 3.19620800e+01, 2.18175774e+01,
                    -3.43563194e+01, -3.16794373e+02, -1.14215515e+02, -1.60282751e+03,
                    -2.34029472e-01, -5.03993154e-01, -7.03439653e-01, 7.47513914e+00,
                    2.69293461e+01, -3.57804832e+01, -3.46590729e+02, -1.82277844e+03,
                    -1.72432056e+03, -1.27731073e+00, -2.57041603e-01, -3.78725111e-01,
                    9.31902599e+00, 2.67350693e+01, -3.00858269e+01, -2.16260269e+02,
                    2.93784943e+02, -5.63845581e+02, -2.28240028e-01, -2.25193590e-01,
                    -6.19725347e-01, 7.64884233e+00, 2.62607403e+01, -3.29219246e+01,
                    -3.30930664e+02, -1.08913989e+03, -1.42398950e+03, -9.96334553e-01,
                    -9.73807927e-03, -2.20419630e-01, 2.81873083e+00, 2.35137424e+01,
                    -3.13092594e+01, 2.39130508e+02, -1.68099585e+03, -2.52542114e+03,
                    -1.48842512e-02, -1.63395017e-01, -4.47753400e-01, 1.08532734e+01,
                    2.09372749e+01, -4.61218376e+01, 1.10423950e+03, -1.25824939e+03,
                    -3.75632471e+03, -1.44030523e+00, 4.82494123e-02, 7.91183561e-02,
                    5.89785719e+00, 2.82478294e+01, -7.24016800e+01, -9.28175781e+02,
                    9.93521667e+02, 3.95115112e+03, -3.51719826e-01, -5.98796785e-01,
                    -2.33891919e-01, 5.38646889e+01, -3.33873701e+00, -2.15550632e+01])


    stds = np.array([6945.848, 2686.427, 2871.097, 32.93815, 37.17426, 27.305304,
                    17.970592, 44.100662, 27.643883, 7298.917, 2551.3857, 3329.0312,
                    51.304676, 49.43646, 31.335526, 16.893368, 47.71109, 32.661713,
                    6938.176, 2715.3987, 3296.7, 44.91464, 41.5619, 31.240667,
                    22.5143, 53.592587, 46.943756, 7372.4243, 2737.9797, 3452.1938,
                    49.012985, 50.870693, 31.164236, 19.169243, 41.900204, 35.39268,
                    7077.101, 2801.966, 3288.3171, 43.516663, 43.00801, 30.701008,
                    23.039852, 70.481316, 48.276485, 5794.938, 3421.688, 4203.817,
                    44.632915, 45.571747, 34.52821, 23.115152, 35.290333, 37.143223,
                    6662.959, 4907.151, 4953.2593, 33.83719, 76.69994, 47.600132,
                    38.09727, 62.66252, 63.31026, 6549.3613, 2293.787, 2729.528,
                    15.296869, 30.679577, 17.03959, 15.159824, 43.69038, 22.054604])

    while True:
        start_time = time.time()

        np.zeros((72))

        for j, k in enumerate(sorted(imu_buff_dict.keys())):
            if imu_buff_dict[k] is not None:
                imu_buff_downsampled[j*9:(j+1)*9] = imu_buff_dict[k] 

        imu_buff_downsampled = (imu_buff_downsampled - means) / stds

        #print(imu_buff_downsampled)
        #print('\n')

        if thread_killer:
            break

        # Control the frame rate
        elapsed_time = time.time() - start_time
        #print(time.time())
        if elapsed_time < interval:
            time.sleep(interval - elapsed_time)

    print(f'Stopping thread {inspect.stack()[0][3]}...')



def capture_video():
    """Capture video frames from the specified video source and add them to the queue."""
    global videoOn1, videoOn2, thread_killer, frame_buff_0, frame_buff_1, fps
    vid1 = cv2.VideoCapture(0, cv2.CAP_V4L)
    vid2 = cv2.VideoCapture(2, cv2.CAP_V4L)
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
    
    while vid1.isOpened() and vid2.isOpened():
        videoOn1, videoOn2 = True, True
        # print(f'vid1: {vid1.isOpened()} vid2: {vid2.isOpened()}')

        start_time = time.time()

        if  vid1:
            ret1, frame1 = vid1.read()
        else:
            continue
        if  vid2:
            ret2, frame2 = vid2.read()
        else:
            continue

        if ret1:
            #frame_buff_0 = frame1
            cv2.imshow(f'frame_{1}', frame1)
        else:
            print('Possible problem in camera1 driver')
            continue
        if ret2:
            #frame_buff_1 = frame2
            cv2.imshow(f'frame_{2}', frame2)
        else:
            print('Possible problem in camera2 driver')
            continue

        #if ret1 and ret2:    
        #    cv2.imshow(f'frame_{1}', frame_buff_0)
        #    cv2.imshow(f'frame_{2}', frame_buff_1)
        #else:
        #    print('Possible problem in camera driver')

        #print(frame_buff_0.shape)
        frame_buff_0 = transform1(frame1)
        #print(frame_buff_0.shape)
        #print('\n')

        frame_buff_1 = transform2(frame2)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if thread_killer:
            break

        # Control the frame rate
        elapsed_time = time.time() - start_time
        if elapsed_time < interval:
            time.sleep(interval - elapsed_time)

    videoOn1, videoOn2 = False, False
    
    vid1.release()
    vid2.release()
    cv2.destroyAllWindows()
    frame_queue1 = None  # Send stop signal to saving thtread
    frame_queue2 = None  # Send stop signal to saving thread
    print(f'Stopping thread {inspect.stack()[0][3]}...')



def online_classification():
    global thread_killer, frame_buff_0, frame_buff_1, experiment_actions, global_idx, imu_buff_downsampled, is_active_classifier, current_action, publisher, fps

    nhead = 16
    num_encoder_layers = 2
    dim_feedforward = 256
    intermediate_dim = 64
    pixel_dim = 224
    patch_size = 56
    max_time = 90
    n_features = 72

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    action_names = ['linger', 'massaging', 'patting', 
                'pinching', 'press', 'pull', 
                'push', 'rub', 'scratching', 
                'shaking', 'squeeze', 'stroke', 
                'tapping', 'trembling', 'idle']

    model = CHARberoViVit(
                        pixel_dim, 
                        patch_size, 
                        len(action_names), 
                        max_time, 
                        n_features, 
                        nhead, 
                        num_encoder_layers, 
                        dim_feedforward, 
                        intermediate_dim
                        ).to(device)

    print('Model Initialized') 
    base_path = '/home/holab/Desktop/JAIST_Cylinder'
    model.load_state_dict(torch.load(f'{base_path}/best_models/_idle_checkpoint_model_IMUdoubleVideo_0.0001lr_16bs_224px_56ps_FalseAug.pt'))
    print('Model State Loaded') 
    model.eval()
    print('Model in evaluation mode\n')

    # define tensor windows
    #video1_window = torch.zeros((max_time, 1, pixel_dim, pixel_dim)).unsqueeze(0)
    #video2_window = torch.zeros((max_time, 1, pixel_dim, pixel_dim)).unsqueeze(0)
    #imu_window = torch.zeros((max_time, n_features)).unsqueeze(0)
    video1_window = torch.full((1, max_time, 1, pixel_dim, pixel_dim), torch.nan)
    video2_window = torch.full((1, max_time, 1, pixel_dim, pixel_dim), torch.nan)
    imu_window = torch.full((1, max_time, n_features), torch.nan)
    print(f'{video1_window.shape=}, {video2_window.shape=}, {imu_window.shape=}\n')

    video1_window = video1_window.to(device)
    video2_window = video2_window.to(device)
    imu_window = imu_window.to(device)
    batch_length = torch.tensor([max_time]).to(device)
    print('Moved windows to device\n')


    interval = 1.0 / fps

    #buffer_insert_timer = 0
    #just_once = True

    while True:
        if frame_buff_0 is None or frame_buff_1 is None:# and imu_buff_downsampled is not None:
            print('Waiting for buffer completion')
            buffer_insert_timer = time.time()
            continue
        else:
            start_time = time.time()

            # Roll the windows
            video1_window = torch.roll(video1_window, shifts=-1, dims=1)
            video2_window = torch.roll(video2_window, shifts=-1, dims=1)
            imu_window = torch.roll(imu_window, shifts=-1, dims=1)

            # Put the new elements in the windows
            video1_window[:, -1, :, :, :] = frame_buff_0[:, :, :]
            video2_window[:, -1, :, :, :] = frame_buff_1[:, :, :]
            imu_window[:, -1, :] = imu_buff_downsampled[:]


            if not torch.isnan(video1_window).any() and not torch.isnan(video2_window).any() and not torch.isnan(imu_window).any():
                if is_active_classifier:
                    #if just_once:
                    #    print(time.time() - buffer_insert_timer)
                    #    just_once = False
                    #    exit()
                    outputs = model(video1_window, video2_window, imu_window, batch_length)

                    outputs = F.softmax(outputs, dim=1)
                    _, predicted = torch.max(outputs, 1)
                    if outputs[0][predicted.item()] > 0.0:
                            output_labels = predicted
                    else:
                        output_labels = 14
                
                    action_performed_long_enough = update_action_timing_window(output_labels)
                    #print(f'.....{action_names[output_labels]}: {output_labels}')
                    if action_performed_long_enough:
                        print(f'.....{action_names[output_labels]}: {output_labels}')
                        #current_action = None
                        # Publish Ros message to start the robot
                        if action_names[output_labels] == experiment_actions[global_idx]:
                            publisher.publish(True)
                            is_active_classifier = False

                            #global_idx = global_idx+1
                            if global_idx < 4:
                                global_idx += 1
                            else:
                                global_idx = 0

            else:
                print('window not full... waiting')
                
            if thread_killer:
                break

            # Control the frame rate
            elapsed_time = time.time() - start_time
            #print(time.time())
            if elapsed_time < interval:
                time.sleep(interval - elapsed_time)



    print(f'Stopping thread {inspect.stack()[0][3]}...')



def main():
    global driver_threads, trial, action, hand
    
    
    print(imu_sensor_names)
    print(sorted(imu_sensor_names))

    # Create threads for capturing video
    video_capture_thread = threading.Thread(target=capture_video, args=())

    # Create thread for IMUs
    imu_thread = threading.Thread(target=IMUs_driver, args=())  
    imu_downsampling_thread = threading.Thread(target=IMUs_downsampler, args=())

    # Create thread for classification
    classification_thread = threading.Thread(target=online_classification, args=())

    driver_threads.extend([
                            video_capture_thread,
                            imu_thread,
                            imu_downsampling_thread,
                            #classification_thread
                         ])


    signal.signal(signal.SIGINT, signal_handler)

    # Start driver threads
    for thread in driver_threads:
        thread.start()

    input('Press START to start the classifier!!!')
    classification_thread.start()
    
    # Wait input before recording
    input('Press START to stop everything!!!')
    #time.sleep(60)
    signal_handler(None, None)


    for thread in driver_threads:
        thread.join()
        print(f'Thread {thread.name} stopped.')

    print('All threads stopped.')


def activate_classifier(msg):
    global is_active_classifier
    is_active_classifier = True


if __name__ == "__main__":
    # Define global variables
    imu_sensor_names = ['Wrist', 'Thumb_Meta', 'Thumb_Distal', 'Index_Proximal', 'Index_Intermediate', 'Middle_Proximal', 'Middle_Intermediate', 'Hand']
    driver_threads = []
    videoOn1, videoOn2 = False, False
    thread_killer = False

    is_active_classifier = False

    # Buffers to hold frames captured from each camera
    frame_buff_0 = None
    frame_buff_1 = None
    imu_buff_dict = {s_name:None for s_name in imu_sensor_names}
    imu_buff_downsampled = torch.zeros((72))
    #print(sorted(imu_buff_dict.keys()))
    #exit()
    hand = 'left'
    fps = 29.0

    #            ['linger', 'massaging', 'patting', 
    #            'pinching', 'press', 'pull', 
    #            'push', 'rub', 'scratching', 
    #            'shaking', 'squeeze', 'stroke', 
    #            'tapping', 'trembling', 'idle']

    experiment_actions = ['linger', 'massaging', 'patting', 'pinching', 'press']
    experiment_actions = ['pull', 'squeeze', 'rub', 'scratching', 'shaking']
    experiment_actions = ['trembling', 'tapping', 'stroke', 'massaging', 'press']
    experiment_actions = ['patting', 'squeeze', 'stroke', 'pull', 'pinching']
    experiment_actions = ['linger', 'rub', 'scratching', 'shaking', 'trembling']

    global_idx = 0 

    rospy.init_node('online_classification', anonymous=True)
    publisher = rospy.Publisher('/StartNextAction', Bool, queue_size = 10)
    rospy.Subscriber('/ActivateClassifier', Bool, activate_classifier, queue_size = 2)

    # Thresholding variables
    current_action = None
    action_start_time = None
    action_threshold = 0.3  # Threshold in seconds
    action_durations = {}
    buffer = torch.full((15,), torch.nan)
    main()

   
