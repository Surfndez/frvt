#include "FaceClassifier.h"
#include "SsdFaceDetector.h"

using namespace FRVT_11;

FaceClassifier::FaceClassifier(const std::string &configDir)
{
    mFaceDetector = std::make_shared<SsdFaceDetector>(configDir, "/facessd_mobilenet_v2_dm100_swish_128x128_wider_filter20_0-509586", 128);
}

bool
FaceClassifier::classify(const cv::Mat& image, const std::vector<float>& features) const
{
    std::vector<Rect> rects = mFaceDetector->Detect(image);
    return rects.size() > 0;
}
