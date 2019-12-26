#ifndef FACERECOGNIZER_H_
#define FACERECOGNIZER_H_

#include <opencv2/core.hpp>

namespace FRVT_11 {
    class FaceRecognizer {
public:
    virtual ~FaceRecognizer() {}

    virtual std::vector<float> Infer(const cv::Mat& image) const = 0;
};
}

#endif /* FACERECOGNIZER_H_ */
