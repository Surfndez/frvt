#include "FaceClassifier.h"
#include "SsdFaceDetector.h"

using namespace FRVT_11;

const float MIN_FEATURES_NORM = 20.f;
const int MIN_LANDMARKS_SCALE = 20;

float
CalcFeaturesNorm(std::vector<float> features)
{
    cv::Mat f(512, 1, CV_32F, features.data());
    float norm = cv::norm(f);
    return norm;
}

int
CalcLandmarksScale(const std::vector<int>& landmarks)
{
    std::vector<int> xPoints;
    std::vector<int> yPoints;
    for (int i = 0; i < 10; i += 2) {
        xPoints.push_back(landmarks[i]);
        yPoints.push_back(landmarks[i+1]);
    }

    int xMin = *std::min_element(xPoints.begin(), xPoints.end());
    int xMax = *std::max_element(xPoints.begin(), xPoints.end());
    int yMin = *std::min_element(yPoints.begin(), yPoints.end());
    int yMax = *std::max_element(yPoints.begin(), yPoints.end());
     
    int w = (xMax - xMin);
    int h = (yMax - yMin);

    return std::max(w, h);
}

FaceClassifier::FaceClassifier(const std::string &configDir)
{
    mFaceDetector = std::make_shared<SsdFaceDetector>(configDir, "/facessd_mobilenet_v2_dm100_swish_128x128_wider_filter20_0-509586", 128);
}

bool
FaceClassifier::classify(const cv::Mat& image, std::vector<int> landmarks, const std::vector<float>& features) const
{
    // if (CalcFeaturesNorm(features) < MIN_FEATURES_NORM)
    // {
    //     return false;
    // }

    // if (CalcLandmarksScale(landmarks) < MIN_LANDMARKS_SCALE)
    // {
    //     return false;
    // }

    // std::vector<Rect> rects = mFaceDetector->Detect(image);
    // if (rects.size() == 0)
    // {
    //     return false;
    // }

    return true;
}
