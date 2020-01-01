#ifndef FACE_CLASSIFIER_H
#define FACE_CLASSIFIER_H

#include <vector>

#include "FaceDetector.h"

namespace FRVT_11 {
    class FaceClassifier
    {
public:
    FaceClassifier(const std::string &configDir);

    bool classify(const cv::Mat& image, std::vector<int> landmarks, const std::vector<float>& features) const;

private:
    std::shared_ptr<FaceDetector> mFaceDetector;
};
}

#endif /* FACE_CLASSIFIER_H */