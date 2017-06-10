import cv2
import sys
import pyautogui
import numpy as np


def main():
    pyautogui.FAILSAFE = False
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

    is_first_frame = True
    previous_x = 0
    previous_y = 0
    while True:
        ok, frame = video.read()
        if not ok: break

        if is_first_frame:
            largest_area = 0
            gaussian_blurred = cv2.GaussianBlur(frame, (5, 5), 0) # Kernel size değişebilir noise robust için
            foreground_mask = mog2_bgs.apply(gaussian_blurred)
            _, thresholded = cv2.threshold(foreground_mask, float(70.0), 255, cv2.THRESH_BINARY)
            gradient = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, element)  # sadece dilation a çevrilebilir
            foreground = gradient

            image, contours, hierarchy = cv2.findContours(foreground.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for i, cnt in enumerate(contours):
                area = cv2.contourArea(cnt)
                tmp_size = np.size(frame)
                if not ((500 < area < 1000) or area > tmp_size / 16):  # area aralığı???? çözülmesi gerekiyor
                    if largest_area < area:
                        x, y, w, h = cv2.boundingRect(cnt)
                        if x > 100 and y > 100 and \
                                x + 100 < video.get(cv2.CAP_PROP_FRAME_WIDTH) and y + 100 < video.get(cv2.CAP_PROP_FRAME_HEIGHT):
                            # ekranın belli ???miktar içinde bulması gerek hareketin
                            if w > 100 and h > 100:  # kare bulmamız ve belli bir uzunluktan??? fazla olması çözülmesi gerekiyor
                                bbox = (x, y, w, w)
                                _ = tracker.init(frame, bbox)
                                largest_area = area
                                is_first_frame = False
        else:
            ok, bbox = tracker.update(frame)

            if ok:
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (0, 0, 255))
                print("%s" % str(bbox))
                if (previous_x - 3 < bbox[0] < previous_x + 3) and (previous_y - 3 < bbox[1] < previous_y + 3):
                    # direkt eşit mi? yoksa bir 3?????!! aralığa göre mi??
                    is_first_frame = True
                else:
                    x, y = pyautogui.position()
                    pyautogui.moveTo(x+10*(bbox[0]-previous_x),y+10*(bbox[1]-previous_y))
                    # senstivity!!??!  pyautogui muazzam!
                    previous_x = bbox[0]
                    previous_y = bbox[1]

            else:
                is_first_frame = True
            # if bbox[0] + bbox[2]/2 < 0 or video.get(cv2.CAP_PROP_FRAME_WIDTH) < (bbox[0] + bbox[2]/2) or bbox[1] + bbox[3]/2 < 0 \
                   #     or video.get(cv2.CAP_PROP_FRAME_HEIGHT) < (bbox[1] + bbox[3]/2):
                    # is_first_frame = True

        cv2.imshow("Tracking", frame)

        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break


if __name__ == '__main__':
    main()