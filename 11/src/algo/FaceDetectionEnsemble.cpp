#include "FaceDetectionEnsemble.h"
#include "SsdFaceDetector.h"

#include <iostream>
#include "../algo/TimeMeasurement.h"

using namespace FRVT_11;

FaceDetectionEnsemble::FaceDetectionEnsemble(const std::string &configDir)
{
    mDetectors = {
        std::make_shared<SsdFaceDetector>(configDir, "/ssdlite_mobilenet_v3_large_416x416_fddb_wider_tasqai_noface2_filter10_0-400000", 416),
        // std::make_shared<SsdFaceDetector>(configDir, "/ssdlite_mobilenet_v3_large_224x224_fddb_wider_tasqai_noface2_filter20_steps_0-300000", 224),
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
            //std::cout << rects[0].x1 << " " << rects[0].y1 << " " << rects[0].x2 << " " << rects[0].y2 << std::endl;
            return {rects[0]};
        }
    }
    return {};
}
