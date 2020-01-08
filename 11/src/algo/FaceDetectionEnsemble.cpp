#include "FaceDetectionEnsemble.h"
#include "SsdFaceDetector.h"

using namespace FRVT_11;

FaceDetectionEnsemble::FaceDetectionEnsemble(const std::string &configDir)
{
    mDetectors = {
        std::make_shared<SsdFaceDetector>(configDir, "/ssdlite_mobilenet_v3_large_416x416_fddb_wider_tasqai_noface2_filter10_0-400000", 416),
        std::make_shared<SsdFaceDetector>(configDir, "/facessd_mobilenet_v2_dm100_swish_512x512_wider_filter10_0-527691", 512)
    };
}

FaceDetectionEnsemble::~FaceDetectionEnsemble() {}

std::vector<Rect>
FaceDetectionEnsemble::Detect(const ImageData &imageData) const
{
    for (auto& d : mDetectors) {
        std::vector<Rect> rects = d->Detect(imageData);
        if (rects.size() > 0) {
            return {rects[0]};
        }
    }
    return {};
}
