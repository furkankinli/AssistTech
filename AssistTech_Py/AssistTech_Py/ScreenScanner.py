import cv2
import numpy as np
import sys
import pyautogui

def main():
    video = cv2.VideoCapture(0)
    while 1:
        read, frame = video.read()
        cv2.clipLine(frame, (60, 70))
        cv2.imshow('DesTek', frame)

        k = cv2.waitKey(1) & 0xff
        if k == 27:
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    main()