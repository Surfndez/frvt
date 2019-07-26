#include "FaceDetectionEnsemble.h"
#include "SsdFaceDetector.h"

#include <iostream>
#include "../algo/TimeMeasurement.h"

using namespace FRVT_11;

FaceDetectionEnsemble::FaceDetectionEnsemble(const std::string &configDir)
{
    mDetectors = {
        std::make_shared<SsdFaceDetector>(configDir, "/fd_tf_dm100_352_0-408944", 352), // facessd_mobilenet_v2_dm100_352_0-408944
        std::make_shared<SsdFaceDetector>(configDir, "/fd_tf_dm100_352_0-277768", 352), // facessd_mobilenet_v2_dm100_352_0-277768
    };
}

FaceDetectionEnsemble::~FaceDetectionEnsemble() {}

std::vector<Rect>
FaceDetectionEnsemble::Detect(const ImageData &imageData) const
{
    for (auto& d : mDetectors) {
        //auto t = TimeMeasurement();
        std::vector<Rect> rects = d->Detect(imageData);
        //std::cout << "Face detection: "; t.Test();
        if (rects.size() > 0) {
            return {rects[0]};
        }
    }
    return {};
}
