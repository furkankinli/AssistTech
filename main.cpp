#include <iostream>
#include <opencv/cv.hpp>

using namespace std;
using namespace cv;

Mat initialFrame;
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

    while (true) {
        if (!(stream.read(initialFrame)))
            break;

        Mat blurred, mogged, thresed, morphed;
        Mat element = getStructuringElement(MORPH_RECT, Size(7, 7));

        GaussianBlur(initialFrame, blurred, Size(5, 5), 0, 0);
        ptrMOG2->apply(blurred, mogged);
        threshold(mogged, thresed, 70.0f, 255, THRESH_BINARY);
        morphologyEx(thresed, morphed, MORPH_OPEN, element);

        vector<vector<Point> > contours;
        findContours(morphed.clone(), contours, CV_RETR_TREE, CV_CHAIN_APPROX_TC89_KCOS);

        Rect bounding_rect;

        for (int i = 0; i < contours.size(); i++) {
            if (contours[i].size() > 50 && contours[i].size() < 300) {
                bounding_rect = boundingRect(contours[i]);
                rectangle(initialFrame, bounding_rect, Scalar(0, 0, 255));
                Point center = (bounding_rect.br() + bounding_rect.tl()) * 0.5;
                circle(initialFrame, center, 1, Scalar(0, 255, 0), 1, 8, 0);
            }
        }
        imshow("Morphed Frame", morphed);
        imshow("Frame", initialFrame);

        if (waitKey(30) == 27) {
            cout << "Key interrupted." << endl;
            break;
        }
    }

    return 0;
}
