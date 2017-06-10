import cv2
import sys 
import ctypes
import numpy as np


def main():
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mog2_bgs = cv2.createBackgroundSubtractorMOG2()
    cv2.ocl.setUseOpenCL(False)
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
    previous_x = 0
    previous_y = 0
    bbox = 0 # python  <3
    while True:
        ok, frame = video.read()
        if not ok: break

        if is_first_frame:
            largest_area = 0
            gaussian_blurred = cv2.GaussianBlur(frame, (5, 5), 0) # Kernel size de?i?ebilir noise robust i�in
            foreground_mask = mog2_bgs.apply(gaussian_blurred)
            _, thresholded = cv2.threshold(foreground_mask, float(70.0), 255, cv2.THRESH_BINARY)
            gradient = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, element)  #sadece dilation a �evrilebilir
            foreground = gradient

            image, contours, hierarchy = cv2.findContours(foreground.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for i, cnt in enumerate(contours):
                area = cv2.contourArea(cnt)
                tmp_size = np.size(frame)
                if not ((1000 < area < 3000) or area > tmp_size / 8): # area aral??? ��z�lmesi gerekiyor
                    if largest_area < area:
                        x, y, w, h = cv2.boundingRect(cnt)
                        if w > 100 and h > 100: # kare bulmam?z ve belli bir uzunluktan fazla olmas? ��z�lmesi gerekiyor
                            bbox = (x, y, w, w)
                            previous_y = y
                            previous_x = x
                            largest_area = area
                            is_first_frame = False
        else:
            tracker = cv2.Tracker_create("KCF")
            ok = tracker.init(frame, bbox)
            ok, bbox = tracker.update(frame)

            if ok:
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (0, 0, 255))
                if (previous_x - 3 < bbox[0] < previous_x + 3) and (previous_y - 3 < bbox[1] < previous_y + 3): # direkt e?it mi? yoksa bir aral??a g�re mi??
                    is_first_frame = True
                    previous_x = 0
                    previous_y = 0
                else:
                    previous_x = bbox[0]
                    previous_y = bbox[1]

                if bbox[0] + bbox[2]/2 < 0 or video.get(cv2.CAP_PROP_FRAME_WIDTH) < (bbox[0] + bbox[2]/2) or bbox[1] + bbox[3]/2 < 0 \
                                        or video.get(cv2.CAP_PROP_FRAME_HEIGHT) < (bbox[1] + bbox[3]/2):
                    is_first_frame = True
                    previous_x = 0
                    previous_y = 0


        cv2.imshow("Tracking", frame)

        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break


if __name__ == '__main__':
    main()