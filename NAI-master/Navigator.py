import os
import sys
import cv2
import time
import autopy
import logging as log
from collections import namedtuple
from argparse import ArgumentParser
from inference import Network
import numpy as np
import dlib
import tkinter as tk
from math import hypot
import pyautogui

CONFIDENCE = 0.7
TARGET_DEVICE = 'CPU'
accepted_devices = ['CPU', 'GPU', 'MYRIAD', 'HETERO:FPGA,CPU', 'HDDL']
SENTIMENT_LABEL = ['neutral', 'happy', 'sad', 'surprise', 'anger']
is_async_mode = True
KEEP_RUNNING = True
DELAY = 5
RUN = False

def draw(mode_type,frame, left1_sq, right1_sq, left_bottom_point, left_top_point, landmarks, lcenter_top, lcenter_bottom, right_bottom_point, left_sq, right_sq, right_top_point, rcenter_top,  rcenter_bottom, xmax, xmin,  gaze_vector, left_midpoint, right_midpoint):
    if mode_type == 'std':
        frame = cv2.line(frame, left_bottom_point, left_top_point,(255,0,255),2)
        frame = cv2.line(frame, lcenter_top, lcenter_bottom, (255,0,255),2)
        frame = cv2.line(frame, right_bottom_point, right_top_point, (255,0,255), 2)
        frame = cv2.line(frame, rcenter_top, rcenter_bottom, (255,0,255),2)
        arrow_length = int(0.3 * xmax-xmin)
        gaze_arrow_left = (int(arrow_length * - gaze_vector[0][0] + left_midpoint[0]), int(arrow_length * gaze_vector[0][1] + left_midpoint[1]))
        gaze_arrow_right = (int(arrow_length * -  gaze_vector[0][0] + right_midpoint[0]), int(arrow_length * gaze_vector[0][1] + right_midpoint[1]))
        frame = cv2.arrowedLine(frame, left_midpoint, gaze_arrow_left, (255, 255, 255), 2)
        frame = cv2.arrowedLine(frame, right_midpoint, gaze_arrow_right, (255, 255, 255), 2)
        frame = cv2.line(frame, (landmarks.part(48).x, landmarks.part(48).y), (landmarks.part(60).x, landmarks.part(60).y), (255,0,255),2)
        frame = cv2.line(frame, (landmarks.part(60).x, landmarks.part(60).y), (int((landmarks.part(67).x+landmarks.part(61).x)/2), int((landmarks.part(67).y+landmarks.part(61).y)/2)), (209,206,0),2)
        frame = cv2.line(frame, (int((landmarks.part(67).x+landmarks.part(61).x)/2), int((landmarks.part(67).y+landmarks.part(61).y)/2)), (int((landmarks.part(66).x+landmarks.part(62).x)/2), int((landmarks.part(66).y+landmarks.part(62).y)/2)), (209,206,0),2)
        frame = cv2.line(frame, (int((landmarks.part(66).x+landmarks.part(62).x)/2), int((landmarks.part(66).y+landmarks.part(62).y)/2)), (int((landmarks.part(65).x+landmarks.part(63).x)/2), int((landmarks.part(65).y+landmarks.part(63).y)/2)), (209,206,0),2)
        frame = cv2.line(frame, (int((landmarks.part(65).x+landmarks.part(63).x)/2), int((landmarks.part(65).y+landmarks.part(63).y)/2)), (landmarks.part(64).x, landmarks.part(64).y), (209,206,0),2)
        frame = cv2.line(frame, (landmarks.part(64).x, landmarks.part(64).y), (landmarks.part(54).x, landmarks.part(54).y), (209,206,0),2)
        for i in range(48,55):
            frame = cv2.line(frame, (landmarks.part(i).x, landmarks.part(i).y), (landmarks.part(i+1).x, landmarks.part(i+1).y), (209,206,0),2)
        frame = cv2.line(frame, (landmarks.part(48).x, landmarks.part(48).y), (landmarks.part(59).x, landmarks.part(59).y), (209,206,0),2)
        for i in range(54,60):
            frame = cv2.line(frame, (landmarks.part(i).x, landmarks.part(i).y), (landmarks.part(i-1).x, landmarks.part(i-1).y), (209,206,0),2)
    if mode_type == 'nar':
        try:
            filename = 'shringan.png'
            oriimg = cv2.imread(filename,-1)
            newimg = cv2.resize(oriimg,(int(landmarks.part(38).x-landmarks.part(41).x)+4,int(landmarks.part(41).y-landmarks.part(38).y)+4))
            newimg2 = cv2.resize(oriimg,(int(landmarks.part(44).x-landmarks.part(47).x)+4,int(landmarks.part(47).y-landmarks.part(44).y)+4))
            y1, y2 = landmarks.part(38).y-2, landmarks.part(41).y+2
            x1, x2 = landmarks.part(41).x-2, landmarks.part(38).x+2

            yo, yt = landmarks.part(44).y-2, landmarks.part(47).y+2
            xo, xt = landmarks.part(47).x-2, landmarks.part(44).x+2

            alpha_s = newimg[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s

            alpha_st = newimg2[:, :, 3] / 255.0
            alpha_lt = 1.0 - alpha_st

            for c in range(0, 3):
                frame[y1:y2, x1:x2, c] = (alpha_s * newimg[:, :, c] +
                                        alpha_l * frame[y1:y2, x1:x2, c])
                frame[yo:yt, xo:xt, c] = (alpha_st * newimg2[:, :, c] +
                                        alpha_lt * frame[yo:yt, xo:xt, c])
            filename1 = 'head1.png'
            oriimg1 = cv2.imread(filename1,-1)
            bottom_right = landmarks.part(17).x, landmarks.part(19).y 
            top_left = landmarks.part(26).x, landmarks.part(24).y - ((landmarks.part(29).y-landmarks.part(27).y)+(landmarks.part(28).y-landmarks.part(27).y))
            newimg3 = cv2.resize(oriimg1,(int(top_left[0]-bottom_right[0]),int(bottom_right[1]-top_left[1])))
            alpha_s3 = newimg3[:, :, 3] / 255.0
            alpha_l3 = 1.0 - alpha_s3
            for c in range(0, 3):
                frame[top_left[1]:bottom_right[1], bottom_right[0]:top_left[0], c] = (alpha_s3 * newimg3[:, :, c] +
                                        alpha_l3 * frame[top_left[1]:bottom_right[1], bottom_right[0]:top_left[0], c])
            frame = cv2.line(frame, (landmarks.part(4).x+10, landmarks.part(4).y), (landmarks.part(48).x-10, landmarks.part(48).y), (30,105,210),3)
            frame = cv2.line(frame, (landmarks.part(54).x+10, landmarks.part(54).y), (landmarks.part(12).x-10, landmarks.part(12).y), (30,105,210),3)
            frame = cv2.line(frame, (landmarks.part(3).x+15, landmarks.part(3).y), (landmarks.part(49).x-15, landmarks.part(49).y), (30,105,210),3)
            frame = cv2.line(frame, (landmarks.part(53).x+15, landmarks.part(53).y), (landmarks.part(13).x-15, landmarks.part(13).y), (30,105,210),3)
            frame = cv2.line(frame, (landmarks.part(2).x+30, int((landmarks.part(2).y+landmarks.part(3).y)/2)), (landmarks.part(50).x-30, int((landmarks.part(50).y+landmarks.part(49).y)/2)), (30,105,210),3)
            frame = cv2.line(frame, (landmarks.part(52).x+30, int((landmarks.part(52).y+landmarks.part(53).y)/2)), (landmarks.part(14).x-30, int((landmarks.part(14).y+landmarks.part(13).y)/2)), (30,105,210),3)
        except:
            print('error')
    return frame

def face_detection(res, initial_wh):
    global CONFIDENCE
    faces = []

    for obj in res[0][0]:
        if obj[2] > CONFIDENCE:
            if obj[3] < 0:
                obj[3] = -obj[3]
            if obj[4] < 0:
                obj[4] = -obj[4]
            xmin = int(obj[3] * initial_wh[0])
            ymin = int(obj[4] * initial_wh[1])
            xmax = int(obj[5] * initial_wh[0])
            ymax = int(obj[6] * initial_wh[1])
            faces.append([xmin, ymin, xmax, ymax])
    return faces

def main():
    global DELAY
    global KEEP_RUNNING
    global TARGET_DEVICE
    global RUN
    global is_async_mode
    mode = 'std'
    input_stream = 0
    cap = cv2.VideoCapture(input_stream)
    if not cap.isOpened():
        print("Cap wasn't open")
        return -1
    if input_stream:
        cap.open(input_stream)
        DELAY = 1000 / cap.get(cv2.CAP_PROP_FPS)
    cur_request_id = 0
    next_request_id = 1
    infer_network = Network()
    infer_network_pose = Network()
    infer_network_gaze = Network()
    plugin, (n_fd, c_fd, h_fd, w_fd) = infer_network.load_model('./face_detection/face_detection.xml', TARGET_DEVICE, 1, 1, 2,)
    n_hp, c_hp, h_hp, w_hp = infer_network_pose.load_model('./pose/pose.xml',
                                                           TARGET_DEVICE, 1,
                                                           3, 2,
                                                           None, plugin)[1]
    infer_network_gaze.load_model('./gaze_estimation/gaze.xml',
                                                           TARGET_DEVICE, 3,
                                                           1, 2,
                                                           None, plugin)
    predictor = dlib.shape_predictor(r"./landmarks/shape_predictor_68_face_landmarks.dat")
    
    if is_async_mode:
        print("Application running in async mode...")
    else:
        print("Application running in sync mode...")
    while cap.isOpened:
        ret, frame = cap.read()
        if not ret:
            KEEP_RUNNING = False
            break
        if frame is None:
            KEEP_RUNNING = False
            log.error("ERROR! blank FRAME grabbed")
            break
        key = cv2.waitKey(1) & 0xFF
        initial_wh = [cap.get(3), cap.get(4)]
        in_frame_fd = cv2.resize(frame, (w_fd, h_fd))
        # Change data layout from HWC to CHW
        in_frame_fd = in_frame_fd.transpose((2, 0, 1))
        in_frame_fd = in_frame_fd.reshape((n_fd, c_fd, h_fd, w_fd))

        key_pressed = cv2.waitKey(int(DELAY))
        if key == ord('n'):
            mode = 'nar'
        if is_async_mode:
            infer_network.exec_net(next_request_id, in_frame_fd)
        else:
            infer_network.exec_net(cur_request_id, in_frame_fd)
        if infer_network.wait(cur_request_id) == 0:
            res = infer_network.get_output(cur_request_id)
            faces = face_detection(res, initial_wh)
            if len(faces) == 1:
                for res_hp in faces:
                    xmin, ymin, xmax, ymax = res_hp
                    head_pose = frame[ymin:ymax, xmin:xmax]
                    in_frame_hp = cv2.resize(head_pose, (w_hp, h_hp))
                    in_frame_hp = in_frame_hp.transpose((2, 0, 1))
                    in_frame_hp = in_frame_hp.reshape((n_hp, c_hp, h_hp, w_hp))

                    infer_network_pose.exec_net(cur_request_id, in_frame_hp)
                    infer_network_pose.wait(cur_request_id)

                    angle_y_fc = infer_network_pose.get_output(0, "angle_y_fc")[0]
                    angle_p_fc = infer_network_pose.get_output(0, "angle_p_fc")[0]
                    angle_r_fc = infer_network_pose.get_output(0, "angle_r_fc")[0]
                    head_pose_angles = np.array([angle_y_fc , angle_p_fc, angle_r_fc], dtype = 'float32')
                    head_pose_angles = head_pose_angles.transpose()
                    
                    frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    selection = (xmin,ymin,xmax,ymax)
                    face = dlib.rectangle(*selection)
                    landmarks = predictor(frame2, face)

                    def midpoint(p1, p2):
                        return int((p1.x + p2.x)/2), int((p1.y+p2.y)/2)
                    try:
                        right_bottom_point = (landmarks.part(36).x, landmarks.part(36).y)
                        right_top_point = (landmarks.part(39).x, landmarks.part(39).y)

                        rcenter_top = midpoint(landmarks.part(37), landmarks.part(38))
                        rcenter_bottom = midpoint(landmarks.part(41), landmarks.part(40))

                        h_length = hypot((right_bottom_point[0]-right_bottom_point[0]),(right_top_point[1]-right_bottom_point[1]))
                        v_length = hypot((rcenter_top[0]-rcenter_bottom[0]),(rcenter_top[1]-rcenter_bottom[1]))
                        
                        left_sq = (landmarks.part(36).x-22, landmarks.part(37).y-13)
                        right_sq = (landmarks.part(39).x+11, landmarks.part(40).y+20)
                        right_eye = frame[left_sq[1]:right_sq[1], left_sq[0]:right_sq[0]]
                        right_midpoint = (int((right_bottom_point[0] + right_top_point[0]) / 2), int((right_bottom_point[1] +right_top_point[1]) / 2))
                        in_frame_right_eye = cv2.resize(right_eye, (60, 60))
                        in_frame_right_eye = in_frame_right_eye.transpose((2, 0, 1))
                        in_frame_right_eye = in_frame_right_eye.reshape((1, 3, 60, 60))
                        
                        left_bottom_point = (landmarks.part(42).x, landmarks.part(45).y)
                        left_top_point = (landmarks.part(45).x, landmarks.part(45).y)

                        lcenter_top = midpoint(landmarks.part(43), landmarks.part(44))
                        lcenter_bottom = midpoint(landmarks.part(47), landmarks.part(46))

                        hor_length = hypot((left_bottom_point[0]-left_top_point[0]),(left_bottom_point[1]-left_top_point[1]))
                        ver_length = hypot((lcenter_top[0]-lcenter_bottom[0]),(lcenter_top[1]-lcenter_bottom[1]))
                        left1_sq = (landmarks.part(42).x-11, landmarks.part(43).y-13)
                        right1_sq = (landmarks.part(45).x+22, landmarks.part(46).y+20)
                        left_eye = frame[left1_sq[1]:right1_sq[1], left1_sq[0]:right1_sq[0]]
                        left_midpoint = (int((left_bottom_point[0] + left_top_point[0]) / 2), int((left_bottom_point[1] + left_top_point[1]) / 2))
                        in_frame_left_eye = cv2.resize(left_eye, (60, 60))
                        in_frame_left_eye = in_frame_left_eye.transpose((2, 0, 1))
                        in_frame_left_eye = in_frame_left_eye.reshape((1, 3, 60, 60))
                        
                        infer_network_gaze.exec_net_g(cur_request_id, in_frame_left_eye, in_frame_right_eye, head_pose_angles)
                        infer_network_gaze.wait(cur_request_id)

                        gaze_vector = infer_network_gaze.get_output(0, 'gaze_vector')
                        norm = np.linalg.norm(gaze_vector)
                        gaze_vector = gaze_vector / norm
                        frame = draw(mode,frame, left1_sq, right1_sq, left_bottom_point, left_top_point, landmarks, lcenter_top, lcenter_bottom, right_bottom_point, left_sq, right_sq, right_top_point, rcenter_top,  rcenter_bottom, xmax, xmin,  gaze_vector, left_midpoint, right_midpoint)
                    except:
                        continue
                    if key == ord('r'):
                        RUN = True
                    if key == ord('s'):
                        RUN = False
                    if RUN:
                        if (hor_length/ver_length > 4) and not (h_length/v_length > 4):
                            pyautogui.click(button='left')
                        if (h_length/v_length > 4) and not (hor_length/ver_length > 4):
                            pyautogui.click(button='right')
                        current_position = autopy.mouse.location()
                        x_new = (int(25 * -gaze_vector[0][0] +
                                    current_position[0]),
                                int(25 * -gaze_vector[0][1] +
                                    current_position[1]))
                        try:
                            autopy.mouse.move(x_new[0], x_new[1])
                        except ValueError:
                            continue
        cv2.imshow('Navigation', frame)
        if key_pressed == 27:
            print("Attempting to stop background threads")
            KEEP_RUNNING = False
            break
        if is_async_mode:
            cur_request_id, next_request_id = next_request_id, cur_request_id
        if key == ord("q"):
            break

    infer_network.clean()
    infer_network_pose.clean()
    cap.release()
    cv2.destroyAllWindows()
