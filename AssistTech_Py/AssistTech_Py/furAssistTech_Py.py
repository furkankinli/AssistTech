import cv2
import numpy as np
import sys


def detect_face(frame):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        bbox = (x, y, w, h)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    return bbox


def main():
    tracker = cv2.Tracker_create("KCF")
    c = input("Face detection (1) vs. Foreground information (2):")
    video = cv2.VideoCapture(0)
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mog2_bgs = cv2.createBackgroundSubtractorMOG2()
    first_frame = True
    bbox = (0,0,0,0)

    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    while 1:
        read, frame = video.read()
        frame = cv2.resize(frame,(320, 320))
        if not read:
            print("Cannot read video file")
            sys.exit()
        frame = cv2.flip(frame, 1)

        if first_frame:
            if c == "1":
                bbox = detect_face(frame=frame)
                first_frame = False
            elif c == "2":
                largest_area = 0

                gaussian_blurred = cv2.GaussianBlur(frame, (5, 5), 0)  # Kernel size değişebilir noise robust için
                foreground_mask = mog2_bgs.apply(gaussian_blurred)
                _, thresholded = cv2.threshold(foreground_mask, float(70.0), 255, cv2.THRESH_BINARY)
                gradient = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, element)  # sadece dilation a çevrilebilir
                foreground = gradient

                image, contours, hierarchy = cv2.findContours(foreground.copy(), cv2.RETR_EXTERNAL,
                                                              cv2.CHAIN_APPROX_SIMPLE)

                for i, cnt in enumerate(contours):
                    area = cv2.contourArea(cnt)
                    tmp_size = np.size(frame)
                    if 1000 < area < tmp_size/8:  # area aralığı???? çözülmesi gerekiyor
                        if largest_area < area:
                            x, y, w, h = cv2.boundingRect(cnt)
                            if 150 > w > 50 and 150 > h > 50:
                                print("4")
                                bbox = (x, y, w, w)
                                largest_area = area
                                first_frame = False
            else:
                print("Wrong input!")
        else:
            ok = tracker.init(frame, bbox)
            ok, bbox = tracker.update(frame)

            if ok:
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (0, 0, 255), 2)

        cv2.imshow('AssistTech', frame)

        k = cv2.waitKey(1) & 0xff
        if k == 27:
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    main()
