#include "FanLandmarksDetector.h"

using namespace FRVT_11;

FanLandmarksDetector::FanLandmarksDetector(const std::string &configDir)
{
    // TODO: Load model like in SFD
}

FanLandmarksDetector::~FanLandmarksDetector() {}

std::vector<int>
FanLandmarksDetector::Detect(const ImageData& imageData, const Rect &face) const
{
    std::vector<int> landmarks;
    return landmarks;
}
