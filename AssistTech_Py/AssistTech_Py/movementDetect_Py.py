import sys
import numpy as np
import cv2


def main():
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mog2_bgs = cv2.createBackgroundSubtractorMOG2()
    video = cv2.VideoCapture(0)

    # Contrast strecthing for RGB images
    # ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
    # channels = cv2.split(ycrcb)
    # cv2.equalizeHist(channels[0], channels[0])
    # cv2.equalizeHist(channels[1], channels[1])
    # cv2.equalizeHist(channels[2], channels[2])
    # ycrcb = cv2.merge(channels)
    # frame = cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR)

    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    ok, frame = video.read()
    if not ok:
        print('Cannot read video file')
        sys.exit()

    while True:
        largest_area = 0
        gaussian_blurred = cv2.GaussianBlur(frame, (5, 5), 0)  # Kernel size değişebilir noise robust için
        foreground_mask = mog2_bgs.apply(gaussian_blurred)
        _, thresholded = cv2.threshold(foreground_mask, float(70.0), 255, cv2.THRESH_BINARY)
        gradient = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, element)  # sadece dilation a çevrilebilir
        foreground = gradient

        image, contours, hierarchy = cv2.findContours(foreground.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            tmp_size = np.size(frame)
            if not ((1000 < area < 2000) or area > tmp_size / 8):  # area aralığı???? çözülmesi gerekiyor
                if largest_area < area:
                    x, y, w, h = cv2.boundingRect(cnt)
                    if x > 100 and y > 100 and \
                                            x + 100 < video.get(cv2.CAP_PROP_FRAME_WIDTH) and y + 100 < video.get(
                        cv2.CAP_PROP_FRAME_HEIGHT):
                        # ekranın belli ???miktar içinde bulması gerek hareketin
                        if w > 50 and h > 50:  # kare bulmamız ve belli bir uzunluktan??? fazla olması çözülmesi gerekiyor
                            bbox = (x, y, w, w)
                            largest_area = area
                            is_first_frame = False

        cv2.imshow("Tracking", frame)

        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break

if __name__ == '__main__':
    main()