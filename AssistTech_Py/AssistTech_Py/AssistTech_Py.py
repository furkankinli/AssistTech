import cv2
import sys
import pyautogui


def main():
    pyautogui.FAILSAFE = False
    cv2.ocl.setUseOpenCL(False)
    video = cv2.VideoCapture(0)
    tracker = cv2.Tracker_create("KCF")
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')

    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    previous_x = 0
    previous_y = 0
    previous_w = 0
    face_detector = True

    while True:
        ok, frame = video.read()
        if not ok:
            print('Cannot read video file')
            sys.exit()

        frame = cv2.flip(frame, 1)
        if face_detector:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                if not (previous_y - 3 < y < previous_y + 3 and previous_x - 3 < x < previous_x + 3):
                    frame = cv2.rectangle(frame, (x, y), (x + w, y + w), (255, 0, 0), 2)
                    roi_gray = gray[y:y + h, x:x + w]
                    roi_color = frame[y:y + h, x:x + w]
                    eyes = eye_cascade.detectMultiScale(roi_gray)
                    for (ex, ey, ew, eh) in eyes:
                        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                    x_pos, y_pos = pyautogui.position()
                    pyautogui.moveTo(x_pos + 8 * (x - previous_x), y_pos + 16 * (y - previous_y))
                    previous_x = x
                    previous_y = y
                    previous_w = w
                else:
                    frame = cv2.rectangle(frame, (previous_x, previous_y), (previous_x + previous_w, previous_y + previous_w), (255, 0, 0), 2)


        else:
            bbox = (video.get(cv2.CAP_PROP_FRAME_WIDTH) / 2 - 100, video.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2 - 100, 200, 200)
            ok = tracker.init(frame, bbox)

            ok, bbox = tracker.update(frame)

            if ok:
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (0, 0, 255), 2)
                print("%s" % str(bbox))
                if not (bbox[0] + bbox[2] / 2 < 0 or video.get(cv2.CAP_PROP_FRAME_WIDTH) < (bbox[0] + bbox[2] / 2) or
                                    bbox[1] + bbox[3] / 2 < 0 \
                                or video.get(cv2.CAP_PROP_FRAME_HEIGHT) < (bbox[1] + bbox[3] / 2)):
                    if not (previous_y - 3 < bbox[1] < previous_y + 3 and previous_x - 3 < bbox[0] < previous_x + 3):
                        x_pos, y_pos = pyautogui.position()
                        pyautogui.moveTo(x_pos + 8 * (bbox[0] - previous_x), y_pos + 16 * (bbox[1] - previous_y))
                        # else:
                        # pyautogui.click()
                        # pyautogui.doubleClick()
                        # pyautogui.rightClick()
                        # pyautogui.scroll(-10)

                    previous_x = bbox[0]
                    previous_y = bbox[1]

        cv2.imshow('img', frame)

        k = cv2.waitKey(1) & 0xff
        if k == 27:
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    main()