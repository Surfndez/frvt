#include "FaceDetectionEnsemble.h"
#include "SsdFaceDetector.h"

#include <iostream>
#include "../algo/TimeMeasurement.h"

using namespace FRVT_11;

FaceDetectionEnsemble::FaceDetectionEnsemble(const std::string &configDir)
{
    mDetectors = {
        std::make_shared<SsdFaceDetector>(configDir, "/mobilenet_v3", 196),
        std::make_shared<SsdFaceDetector>(configDir, "/mobilenet_v3", 196),
        // std::make_shared<SsdFaceDetector>(configDir, "/ssdlite_mobilenet_v3_large_320x320_coco_0-26694", 320),
        // std::make_shared<SsdFaceDetector>(configDir, "/facessd_mobilenet_v2_dm100_swish_96_0-525313", 96),
        // std::make_shared<SsdFaceDetector>(configDir, "/fd_tf_dm100_352_0-408944", 352), // facessd_mobilenet_v2_dm100_352_0-408944
        // std::make_shared<SsdFaceDetector>(configDir, "/fd_tf_dm100_512_0-527691", 512), // facessd_mobilenet_v2_dm100_swish_512_0-527691
        // std::make_shared<SsdFaceDetector>(configDir, "/fd_tf_dm100_352_0-277768", 352), // facessd_mobilenet_v2_dm100_352_0-277768
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
