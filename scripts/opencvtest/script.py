from json import load
from posixpath import dirname
import cv2
import os
from torch_model import load_model, make_inference
from time import sleep

MODEL_PATH = '../../model/fully_connected_veri_good.pt'
DIRNAME = os.path.dirname(__file__)


def main():
    model = load_model(os.path.join(DIRNAME, MODEL_PATH))

    vid = cv2.VideoCapture(0)

    b_x = 100
    b_y = 100

    size = 200

    b_width = b_x + size
    b_height = b_y + size
    i = 0
    while(True):
        _, frame = vid.read()
        i += 1
        if i % 10 == 0:
            cv2.rectangle(frame, (b_x, b_y),
                          (b_width, b_height), (255, 0, 0), 2)
            ROI = frame[b_y:b_height, b_x:b_width]
            cv2.imshow('frame2', frame)
            inference = make_inference(model, ROI)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            print(inference)

    vid.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
