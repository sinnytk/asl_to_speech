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

MODEL_PATH = '../../model/custom_model_0.pt'
DIRNAME = os.path.dirname(__file__)

def sayword(Message):
    speech = gTTS(text = Message)
    speech.save('word.mp3')
    playsound('word.mp3')

def main():
    # sayword('hello world')
    model = load_model(MODEL_PATH)

    vid = cv2.VideoCapture(0)

    b_x = 50
    b_y = 50

    size = 224

    b_width = b_x + size
    b_height = b_y + size
    i = 0
    inference=[]
    Message = ''
    while(True):
        _, frame = vid.read()
        i += 1
        cv2.putText(frame, f"{Message}", (45, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (166,247,218), 2, cv2.LINE_AA)

        cv2.rectangle(frame, (b_x, b_y),
                        (b_width, b_height), (255, 0, 0), 2)
        ROI = frame[b_y:b_height, b_x:b_width]

        if i % 3 == 0:
            inference.append(make_inference(model, ROI))
        if i % 30 == 0:
            try:
                annotaion = mode(inference)
                votes = inference.count(annotaion) 
                if votes >=int(0.7*len(inference)):

                    if annotaion == 'space':
                        sayword(Message.lower())
                        Message+=" "
                        
                    if annotaion!='del' and annotaion!='nothing' and annotaion!='space':
                        print(f'{votes} out of {len(inference)} voted for {annotaion} ')

                        Message+=annotaion
                    if annotaion == 'del':
                        print(f'removed:{Message[-1]} from the text')
                        Message=Message[:-1]
                    
                    inference=[]
                    
                else:
                    print(f'current text:{Message.lower()} try again')        
                    cv2.putText(frame, f"{Message}", (45, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,10,10), 2, cv2.LINE_AA)
                    inference=[]

 
            except:
                pass
        cv2.imshow('frame2', frame)
                
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            print(inference)

    vid.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()