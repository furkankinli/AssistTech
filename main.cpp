#include <iostream>
#include <windows.h>
#include <opencv/cv.hpp>
// mftsadasdadadsasdsqwdasdqwdasd
using namespace std;
using namespace cv;

Mat frame;
Ptr<BackgroundSubtractorMOG2> ptrMOG2;

int main() {
    ptrMOG2 = createBackgroundSubtractorMOG2();

    VideoCapture stream(0);
    if (!stream.isOpened()) {
        cout << "Cannot open the video file!" << endl;
        return -1;
    }

    double fps = stream.get(CV_CAP_PROP_FPS);
    cout << "Frame per seconds: " << fps << endl;
    namedWindow("Frame", CV_WINDOW_AUTOSIZE);

    vector<Point> previousContour;
    vector<Point> currentContour;
    int counter = 0;

    while (true) {
        if (!(stream.read(frame)))
            break;

        Mat blurredFrame, moggedFrame, threshedFrame, morphedFrame;
        Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));

        GaussianBlur(frame, blurredFrame, Size(5, 5), 0, 0);
        ptrMOG2->apply(blurredFrame, moggedFrame);
        threshold(moggedFrame, threshedFrame, 70.0f, 255, THRESH_BINARY);
        morphologyEx(threshedFrame, morphedFrame, MORPH_OPEN, element);

        vector<vector<Point> > contours;
        findContours(morphedFrame.clone(), contours, CV_RETR_TREE, CV_CHAIN_APPROX_TC89_KCOS);

        Rect bounding_rect;
        vector<vector<Point> >::iterator it = contours.begin();
        int largest = 0;
        double minDistance = 1000000000;
        Point move;
        while (it != contours.end()) {
            if (it->size() < 100 || it->size() > 400) it = contours.erase(it);
            else {
                if (counter == 0) {
                    vector<Point> point = *it;
                    if (largest < it->size()) currentContour = point;
                } else {
                    vector<Point> point = *it;
                    Moments mu = moments(point, false);
                    Moments prevMu = moments(previousContour, false);
                    Point center = Point2f(mu.m10 / mu.m00, mu.m01 / mu.m00);
                    Point prevCenter = Point2f(prevMu.m10 / prevMu.m00, prevMu.m01 / prevMu.m00);
                    Point diff = center - prevCenter;
                    double distance = norm(diff);

                    if (minDistance > distance) {
                        minDistance = distance;
                        currentContour = point;
                    }
                }
                ++it;
                counter++;
                previousContour = currentContour;
            }
            bounding_rect = boundingRect(currentContour);
            //rectangle(frame, bounding_rect, Scalar(0, 0, 255));
            Point center = (bounding_rect.br() + bounding_rect.tl()) * 0.5;
            circle(frame, center, 1, Scalar( 255, 0, 0), 20, 8, 0);
        }



        imshow("Morphed Frame", morphedFrame);
        imshow("Frame", frame);

        if (waitKey(30) == 27) {
            cout << "Key interrupted." << endl;
            break;
        }
    }
    return 0;
}
