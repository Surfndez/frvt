#include "FaceDetectionEnsemble.h"
#include "SsdFaceDetector.h"

using namespace FRVT_11;

FaceDetectionEnsemble::FaceDetectionEnsemble(const std::string &configDir)
{
    mDetectors = { 
        std::make_shared<SsdFaceDetector>(configDir, "/fd_tf_dm100_352_0-277768", 352) // facessd_mobilenet_v2_dm100_352_0-277768
    };
}

FaceDetectionEnsemble::~FaceDetectionEnsemble() {}

std::vector<Rect>
FaceDetectionEnsemble::Detect(const ImageData &imageData) const
{
    std::vector<Rect> rects;
    for (auto& d : mDetectors) {
        // std::vector<Rect> rects = d->Detect(imageData);
        // rects.push_back(rects[0]);
        return d->Detect(imageData);
    }
    return rects;
}
