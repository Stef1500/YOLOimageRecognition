from Modules import MouseMoveTo, MouseClick, MouseEvent
import cv2
import numpy as np
from mss import mss
import pydirectinput
from time import sleep
from time import time
import keyboard

from ultralytics import YOLO


def MoveAndClick(x1, y1, x2, y2):
    center_x = int((x1+x2)/2)
    center_y = int((y1+y2)/2)

    print(f"Center x = {center_x}, Center y = {center_y}")

    current_mouse_pos = pydirectinput.position()
    print(f"current_mouse_pos: {current_mouse_pos}")

    next_move = center_x - current_mouse_pos[0], center_y - current_mouse_pos[1]
    MouseMoveTo(next_move[0], next_move[1])
    print(f"Moving mouse to: {next_move}")
    sleep(0.01)

    if ((next_move[0] == 0 or not next_move[0] > 3 and not next_move[0] < -3)
            and (next_move[1] == 0 or not next_move[1] > 3 and not next_move[1] < -3)):
        MouseClick()
        sleep(0.01)

def FpsCounter(frame_count, elapsed_time,image):
    font = cv2.FONT_HERSHEY_SIMPLEX

    fps = frame_count / elapsed_time if elapsed_time > 0 else 0

    cv2.putText(image,f"FPS: {round(fps)} ",(1,25) ,font,0.6,(0,255,0),2,cv2.LINE_AA)

def DrawCenter(x1,y1,x2,y2,image):

    x_center = int((x1+x2)/2)
    y_center = int((y1+y2)/2)


    cv2.circle(image,(x_center,y_center),1,(0,0,255),2)


def Killswitch():
    if keyboard.is_pressed('q'):
        print("KILL IT WITH FIRE!")
        exit(1)

################################################################################
#
# MAIN PROGRAM STARTS HERE
#
#################################################################################

#update current paths as desired
path = r"C:\Users\Stefano B\Desktop\Programing courses FSU\CAP4601\YOLO\yolo8clone"
model_path = path +r"\models\aimlabsbluecirclesAIV2.onnx"
start_time = time()
screen_boundaries = {'top': 0, 'left': 0, 'width': 1920, 'height': 1080}
frame_counter = 0


model = YOLO(model_path)
distance_best = np.inf


with mss() as sct:



    while True:
        Killswitch()
        screen = np.asarray(sct.grab(screen_boundaries))
        screen = cv2.cvtColor(screen, cv2.COLOR_BGRA2BGR)
        #screen = cv2.resize(screen, (1920, 10))

        results = model(screen,conf=0.20,verbose=False)
        image = results[0].plot()
        elapsed_time = time() - start_time
        frame_counter += 1
        FpsCounter(frame_counter, elapsed_time,image)

        screen_center_x = 1920/2
        screen_center_y = 1080/2

        if len(results[0].boxes) > 0:
            last_distance = np.inf
            closest_circle = None
            #Looks for the closes box to the center
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()  # x1,y1 = top left and x2,y2 = bottom right
                conf = box.conf.item()
                DrawCenter(x1, y1, x2, y2, image)

                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                #Calculate distance from the center
                distance = ((center_x - screen_center_x)**2 + (center_y - screen_center_y)**2)**0.5

                #looks for the closes circle
                if distance < last_distance:
                    last_distance = distance
                    closest_circle = (x1, y1, x2, y2)
                    print(f"Closest circle in {x1,y1,x2,y2}")
                print(f"Coordinates  {x1}, {x2}, {y1}, {y2}, Confidence: {round(conf, 2)}")
            #focuses on the closest box
            if closest_circle is not None:
                x1, y1, x2, y2 = closest_circle
                MoveAndClick(x1, y1, x2, y2)
        else:
            #What to do if there are no detections
            MouseMoveTo(15, 0)

        cv2.imshow('screen', image)

        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            cv2.destroyAllWindows()
            break

