#include "FaceClassifier.h"
#include "SsdFaceDetector.h"

using namespace FRVT_11;

float
CalcFeaturesNorm(std::vector<float> features)
{
    cv::Mat f(512, 1, CV_32F, features.data());
    float norm = cv::norm(f);
    return norm;
}

FaceClassifier::FaceClassifier(const std::string &configDir)
{
    mFaceDetector = std::make_shared<SsdFaceDetector>(configDir, "/facessd_mobilenet_v2_dm100_swish_128x128_wider_filter20_0-509586", 128);
}

bool
FaceClassifier::classify(const cv::Mat& image, std::vector<int> landmarks, const std::vector<float>& features) const
{
    std::vector<Rect> rects = mFaceDetector->Detect(image);
    return rects.size() > 0;
}
