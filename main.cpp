#include <iostream>
#include <windows.h>
#include <opencv/cv.hpp>
#include <opencv2/features2d.hpp>

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

    // Setup SimpleBlobDetector parameters.
    SimpleBlobDetector::Params params;

    // Change thresholds
    params.minThreshold = 20;
    params.maxThreshold = 1000;

    // Filter by Area.
    params.filterByArea = true;
    params.minArea = 150;

    // Filter by Circularity
    params.filterByCircularity = true;
    params.minCircularity = 0.1;

    // Filter by Convexity
    params.filterByConvexity = true;
    params.minConvexity = 0.2;

    // Filter by Inertia
    params.filterByInertia = true;
    params.minInertiaRatio = 0.01;

    // Storage for blobs
    vector<KeyPoint> keypoints;

    // Set up detector with params
    Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);

    while(true) {
        if(!stream.read(frame)) break;

        // Detect blobs
        detector->detect(frame, keypoints);

        Mat im_with_keypoints;
        drawKeypoints(frame, keypoints, im_with_keypoints, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

        // Show blobs
        imshow("Keypoints", im_with_keypoints );
        if (waitKey(30) == 27) {
            cout << "Key interrupted." << endl;
            break;
        }
    }
    return 0;
}
