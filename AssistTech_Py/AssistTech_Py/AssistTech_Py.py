import cv2
import numpy as np
import sys
import pyautogui


def detect_face(frame):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = face_cascade.detectMultiScale(gray, 1.4, 3)
    for (x, y, w, h) in faces:
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        bbox = (x, y, w, h)
    return bbox


def is_on_screen(bbox, video):
    if bbox[0] < 0:
        return False
    elif video.get(cv2.CAP_PROP_FRAME_WIDTH) < (bbox[0] + bbox[2]):
        return False
    elif bbox[1] < 0:
        return False
    elif video.get(cv2.CAP_PROP_FRAME_HEIGHT) < (bbox[1] + bbox[3]):
        return False
    return True


def draw_rectangle(bbox, frame):
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    cv2.rectangle(frame, p1, p2, (0, 0, 255), 2)


def get_sensitivity(video, flag, conf=1):
    w_screen = pyautogui.size()[0]
    h_screen = pyautogui.size()[1]
    w_cam = float(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_cam = float(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if flag == "w":
        return conf * (w_cam / w_screen)
    else:
        return conf * (h_cam / h_screen)


def dis(b1, b2):
    print(b1)
    print(b2)
    a = (int(b1[0]))
    b = (int(b2[0]))
    c = (int(b1[1]))
    d = (int(b2[1]))
    return np.sqrt((a - b) ** 2 + (c - d) ** 2)


bl = []


def get_acceleration(blob, conf=5):
    global bl
    i = 1
    if len(bl) < 4:
        bl.append(blob)
    elif blob != bl[-1]:
        i = dis(bl[-4], blob) / (4 * conf)
        bl = bl[-3:]
        bl.append(blob)
    else:
        bl = []
    return i


is_clicked = False
counter = 0


def click(prev, bbox, conf=3):
    global is_clicked
    global counter
    num_of_frame = conf * 3
    print("Number of frames: %s" % num_of_frame)
    if dis(prev, bbox) > 3:  # ????????????????????
        if counter > num_of_frame and not is_clicked:
            # need flags for double click, right click, scrolling, drag drop
            pyautogui.click()
            counter = 0
            print("Tıkladım.")
            is_clicked = True
        if counter == 0:
            is_clicked = False


def move(bbox, prev_blob, x_pos, y_pos, video):
    pyautogui.moveTo(x_pos + 16 * get_acceleration(bbox) * (bbox[0] - prev_blob[0]) * get_sensitivity(video, "w"),
                     y_pos + 24 * get_acceleration(bbox) * (bbox[1] - prev_blob[1]) * get_sensitivity(video, "h"))


def stay_on_screen(prev_bbox, video):
    width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # workspace settings  = 10 ??
    if prev_bbox[0] < 0:
        return (10, prev_bbox[1], prev_bbox[2], prev_bbox[3])
    elif width < (prev_bbox[0] + prev_bbox[2]):
        return (width - prev_bbox[2] - 10, prev_bbox[1], prev_bbox[2], prev_bbox[3])
    elif prev_bbox[1] < 0:
        return (prev_bbox[0], 10, prev_bbox[2], prev_bbox[3])
    elif height < (prev_bbox[1] + prev_bbox[3]):
        return (prev_bbox[0], height - prev_bbox[3] - 10, prev_bbox[2], prev_bbox[3])
    return prev_bbox


def main():
    pyautogui.FAILSAFE = False
    pyautogui.MINIMUM_DURATION = 0.5
    tracker = cv2.Tracker_create("KCF")
    c = input("Face detection (1) vs. Foreground information (2) vs. Simple Blob Detection (3):")
    video = cv2.VideoCapture(0)
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mog2_bgs = cv2.createBackgroundSubtractorMOG2()
    params = cv2.SimpleBlobDetector_Params()

    params.minThreshold = 100
    params.maxThreshold = 2000
    params.filterByArea = True
    params.minArea = 3000
    params.filterByCircularity = False
    params.minCircularity = 0.1
    params.filterByConvexity = False
    params.minConvexity = 0.87
    params.filterByInertia = False
    params.minInertiaRatio = 0.01
    detector = cv2.SimpleBlobDetector_create(params)

    first_frame = True
    bbox = (0, 0, 0, 0)
    prev_blob = 0

    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    global counter

    while 1:
        print("Counter: %s" % counter)
        read, frame = video.read()
        # frame = cv2.resize(frame, (320, 320))
        x_pos, y_pos = pyautogui.position()
        fps = 30  # video.get(cv2.CAP_PROP_FPS)
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
                    if 1000 < area < tmp_size / 8:  # area aralığı???? çözülmesi gerekiyor
                        if largest_area < area:
                            x, y, w, h = cv2.boundingRect(cnt)
                            if w > 280 and h > 280:
                                bbox = (x, y, w, w)
                                largest_area = area
                                first_frame = False
            elif c == "3":
                print(first_frame)
                largest_diameter = 0
                keypoints = detector.detect(frame)
                for i, _ in enumerate(keypoints):
                    if largest_diameter < int(keypoints[i].size):
                        largest_diameter = keypoints[i].size
                        bbox = (int(keypoints[i].pt[0]), int(keypoints[i].pt[1]), int(keypoints[i].size),
                                int(keypoints[i].size))
                if largest_diameter > 80: first_frame = False

            else:
                print("Wrong input!")
        else:
            ok = tracker.init(frame, bbox)

            if prev_blob == 0:
                ok, bbox = tracker.update(frame)
                prev_blob = bbox
            elif is_on_screen(prev_blob, video):
                ok, bbox = tracker.update(frame)
                if abs(bbox[0] - prev_blob[0]) < 4 and abs(bbox[1] - prev_blob[1]) < 4:
                    bbox = prev_blob
                    counter += 1
                elif abs(bbox[0] - prev_blob[0]) < 4 or abs(bbox[1] - prev_blob[1]) < 4:
                    if abs(bbox[0] - prev_blob[0]) < 4:
                        bbox = (prev_blob[0], bbox[1], bbox[2], bbox[3])
                    else:
                        bbox = (bbox[0], prev_blob[1], bbox[2], bbox[3])
                if ok:
                    draw_rectangle(bbox, frame)
                    move(bbox, prev_blob, x_pos, y_pos, video)
                    click(prev_blob, bbox)
                prev_blob = bbox
            else:
                prev_blob = stay_on_screen(prev_blob, video)
                draw_rectangle(prev_blob, frame)
                ok = tracker.init(frame, prev_blob)
                move(bbox, prev_blob, x_pos, y_pos, video)
                click(prev_blob, bbox)

        cv2.imshow('DesTek', frame)

        k = cv2.waitKey(1) & 0xff
        if k == 27:
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    main()
