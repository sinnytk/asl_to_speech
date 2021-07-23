from json import load
from posixpath import dirname
import cv2
import os
import torch
from torch_model import load_model, make_inference
from time import sleep
from statistics import mode
from gtts import gTTS
from playsound import playsound

import mediapipe as mp
def sayword(Message):
    speech = gTTS(text = Message)
    speech.save('word.mp3')
    playsound('word.mp3')

MODEL_PATH = '../../model/resnet18_1.pt'
DIRNAME = os.path.dirname(__file__)
model = load_model(os.path.join(DIRNAME, MODEL_PATH))

mphands = mp.solutions.hands
hands = mphands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

_, frame = cap.read()

h, w, c = frame.shape
i=0
Message = ""
inference = []
while True:
    i+=1
    _, frame = cap.read()
    frame = cv2.flip(frame,1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(framergb)
    hand_landmarks = result.multi_hand_landmarks
    handedness = result.multi_handedness
    if hand_landmarks and handedness[0].classification[0].label == 'Right':
        wrist_landmark = hand_landmarks[0].landmark[0]
        wrist_origin = int(wrist_landmark.x*w), int(wrist_landmark.y*h)
        cv2.rectangle(frame, (wrist_origin[0]-100, wrist_origin[1]-200), (wrist_origin[0]+100, wrist_origin[1]+50), (255, 255, 0), 2)
        ROI = frame[wrist_origin[1]-200:wrist_origin[1]+50,wrist_origin[0]-100:wrist_origin[0]+100]
        # cv2.imshow('ROI',ROI)
        if i % 1 == 0:
            inference.append(make_inference(model, ROI))
        if i % 20 == 0:
            try:
                annotaion = mode(inference)
                votes = inference.count(annotaion) 
                if votes >=int(0.7*len(inference)):

                    if annotaion == 'space':
                        sayword(Message.lower())
                    if annotaion!='del' and annotaion!='nothing':
                        print(f'{votes} out of {len(inference)} voted for {annotaion} ')
                        Message+=annotaion
                    else:
                        Message=Message[:-1]
                    
                    inference=[]
                    
                else:
                    print(f'current text:{Message.lower()}')    
            except:
                pass
        cv2.putText(frame, f"{Message}", (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (166,247,218), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, f"{Message}", (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (51, 51, 255), 2, cv2.LINE_AA)

    cv2.imshow("Frame", frame)

    cv2.waitKey(1)