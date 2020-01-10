#include "FaceDetectionEnsemble.h"
#include "SsdFaceDetector.h"

using namespace FRVT_11;

FaceDetectionEnsemble::FaceDetectionEnsemble(const std::string &configDir)
{
    mDetectors = {
        std::make_shared<SsdFaceDetector>(configDir, "/fd_416_400000", 416), // ssdlite_mobilenet_v3_large_416x416_fddb_wider_tasqai_noface2_filter10_0-400000
        // std::make_shared<SsdFaceDetector>(configDir, "/fd_224_300000", 224), // ssdlite_mobilenet_v3_large_224x224_fddb_wider_tasqai_noface2_filter20_steps_0-300000
        // std::make_shared<SsdFaceDetector>(configDir, "/fd_512_527691", 512) // facessd_mobilenet_v2_dm100_swish_512x512_wider_filter10_0-527691
    };
}

FaceDetectionEnsemble::~FaceDetectionEnsemble() {}

std::vector<Rect>
FaceDetectionEnsemble::Detect(const cv::Mat& constImage) const
{
    for (auto& d : mDetectors) {
        std::vector<Rect> rects = d->Detect(constImage);
        if (rects.size() > 0) {
            return {rects[0]};
        }
    }
    return {};
}
