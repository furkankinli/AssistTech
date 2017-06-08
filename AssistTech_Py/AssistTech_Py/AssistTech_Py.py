import cv2
import sys
import ctypes
import numpy as np


def main():
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mog2_bgs = cv2.createBackgroundSubtractorMOG2()
    cv2.ocl.setUseOpenCL(False)
    tracker = cv2.Tracker_create("KCF")
    video = cv2.VideoCapture(0)

    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    ok, frame = video.read()
    if not ok:
        print('Cannot read video file')
        sys.exit()

    # Uncomment the line below to select a different bounding box
    # bbox = cv2.selectROI(frame, False)

    is_first_frame = True

    while True:
        ok, frame = video.read()
        if not ok: break

        if is_first_frame:
            largest_area = 0
            gaussian_blurred = cv2.GaussianBlur(frame, (7, 7), 0) # Kernel size değişebilir noise robust için
            foreground_mask = mog2_bgs.apply(gaussian_blurred)
            _, thresholded = cv2.threshold(foreground_mask, float(70.0), 255, cv2.THRESH_BINARY)
            gradient = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, element)  #sadece dilation a çevrilebilir
            foreground = gradient

            image, contours, hierarchy = cv2.findContours(foreground.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for i, cnt in enumerate(contours):
                area = cv2.contourArea(cnt)
                tmp_size = np.size(frame)
                if not ((1000 < area < 3000) or area > tmp_size / 8): # area aralığı çözülmesi gerekiyor
                    if largest_area < area:
                        x, y, w, h = cv2.boundingRect(cnt)
                        if w > 50 and h > 50: # kare bulmamız ve belli bir uzunluktan fazla olması çözülmesi gerekiyor
                            bbox = (x, y, w, w)
                            largest_area = area
                            ok = tracker.init(frame, bbox)
                            is_first_frame = False
        else:
            ok, bbox = tracker.update(frame)

            if ok:
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (0, 0, 255))

        cv2.imshow("Tracking", frame)

        k = cv2.waitKey(1) & 0xff
        if k == 27: break


if __name__ == '__main__':
    main()