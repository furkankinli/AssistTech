#include <iostream>
#include <windows.h>
#include <opencv/cv.hpp>

using namespace std;
using namespace cv;

Mat frame;

int main() {
    VideoCapture stream(0);
    if (!stream.isOpened()) {
        cout << "Cannot open the video file!" << endl;
        return -1;
    }

    double fps = stream.get(CV_CAP_PROP_FPS);
    cout << "Frame per seconds: " << fps << endl;
    namedWindow("Frame", CV_WINDOW_AUTOSIZE);

    SimpleBlobDetector::Params params;

    params.minThreshold = 10;
    params.maxThreshold = 1000;

    params.filterByArea = true;
    params.minArea = 200;

    params.filterByCircularity = false;

    params.filterByConvexity = true;
    params.minConvexity = 0.3;

    params.filterByInertia = false;
    params.minInertiaRatio = 0.1;

    params.minDistBetweenBlobs = 20;

    vector<KeyPoint> prevKeyPoints;
    vector<KeyPoint> keypoints;

    Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);
    bool isFirst = true;
    Point center, prevCenter;

    while (true) {
        Mat im_with_keypoints;
        int x_total = 0, y_total = 0;

        if (!stream.read(frame)) break;
        Mat blurred, eq;
        GaussianBlur(frame, blurred, Size(21, 21), 0, 0);

        detector->detect(blurred, keypoints);

        for (int i = 0; i < keypoints.size(); i++) {
            if (!keypoints.empty()) {
                x_total += keypoints[i].pt.x;
                y_total += keypoints[i].pt.y;
            }

            stringstream ss;
            ss << keypoints[i].pt.x;
            string str = ss.str();
            cout << str << endl;
        }

        if (isFirst) {
            center = Point2f(x_total / keypoints.size(), y_total / keypoints.size());
            prevCenter = center;
            isFirst = false;
        } else {
            if (!keypoints.empty()) {
                Point fakeCenter = Point2f(x_total / keypoints.size(), y_total / keypoints.size());
                if (abs(fakeCenter.x - prevCenter.x > 15) || abs(fakeCenter.y - prevCenter.y > 15)) {
                    center = fakeCenter;
                } else {
                    center = prevCenter;
                }
            } else {
                center = Point(-1,-1);

            }
        }

        circle(frame, center, 1, Scalar(255, 0, 0), 20, 8, 0);
        drawKeypoints(frame, keypoints, im_with_keypoints, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

        imshow("Keypoints", im_with_keypoints);
        imshow("asdasdasd", frame);

        if (waitKey(30) == 27) {
            cout << "Key interrupted." << endl;
            break;
        }
    }
    return 0;
}
