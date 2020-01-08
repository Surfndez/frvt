#ifndef FACE_CLASSIFIER_H
#define FACE_CLASSIFIER_H

#include <vector>

#include "FaceDetector.h"

namespace FRVT_11 {
    enum class FaceClassificationResult
    {
        Pass,
        Norm,
        Lscale,
        Liou,
        NoFace,
        Fiou
    };

    class FaceClassifier
    {
public:
    FaceClassifier(const std::string &configDir);

    FaceClassificationResult classify(const cv::Mat& image, const Rect& face, std::vector<int> landmarks, const std::vector<float>& features) const;

private:
    std::shared_ptr<FaceDetector> mFaceDetector;
};
}

#endif /* FACE_CLASSIFIER_H */