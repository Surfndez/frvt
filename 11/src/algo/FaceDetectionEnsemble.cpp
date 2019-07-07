#include "FaceDetectionEnsemble.h"
#include "SsdFaceDetector.h"

#include <iostream>
#include "../algo/TimeMeasurement.h"

using namespace FRVT_11;

FaceDetectionEnsemble::FaceDetectionEnsemble(const std::string &configDir)
{
    mDetectors = { 
        std::make_shared<SsdFaceDetector>(configDir, "/fd_tf_dm100_352_0-277768", 352), // facessd_mobilenet_v2_dm100_352_0-277768
        std::make_shared<SsdFaceDetector>(configDir, "/facessd_mobilenet_v2_dm100_352_0-408944", 352),
        //std::make_shared<SsdFaceDetector>(configDir, "/facessd_res26_352_0-239092", 352)
    };
}

FaceDetectionEnsemble::~FaceDetectionEnsemble() {}

std::vector<Rect>
FaceDetectionEnsemble::Detect(const ImageData &imageData) const
{
    std::vector<std::vector<Rect>> detections;

    for (auto& d : mDetectors) {
        //auto t = TimeMeasurement();
        std::vector<Rect> rects = d->Detect(imageData);
        //std::cout << "Face detection: "; t.Test();
        if (rects.size() > 0.3) {
            detections.push_back(rects);
        }
    }
    
    Rect rect(0, 0, 0, 0, 100);
    for (auto& d : detections) {
        rect.x1 += d[0].x1;
        rect.x2 += d[0].x2;
        rect.y1 += d[0].y1;
        rect.y2 += d[0].y2;
    }

    float num_rects = float(detections.size());
    rect.x1 = int(rect.x1 / num_rects);
    rect.x2 = int(rect.x2 / num_rects);
    rect.y1 = int(rect.y1 / num_rects);
    rect.y2 = int(rect.y2 / num_rects);

    return {rect};
}
