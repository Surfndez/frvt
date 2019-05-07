#include "SphereFaceRecognizer.h"

using namespace FRVT_11;

SphereFaceRecognizer::SphereFaceRecognizer(const std::string &configDir)
{
    std::string landmarksDetectorModelPath = configDir + "/sphereface-sphereface_84_cosineface_nist_bbox_1-3250000.pb";

    // TODO: Load model
}

std::vector<float>
SphereFaceRecognizer::Infer(const ImageData& imageData, const std::vector<int>& landmarks) const
{
    std::vector<float> features;
    return features;
}