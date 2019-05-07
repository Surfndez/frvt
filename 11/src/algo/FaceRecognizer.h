#ifndef FACERECOGNIZER_H_
#define FACERECOGNIZER_H_

#include <vector>

#include "ImageData.h"

namespace FRVT_11 {
    class FaceRecognizer {
public:
    virtual ~FaceRecognizer() {}

    virtual std::vector<float> Infer(const ImageData& imageData, const std::vector<int>& landmarks) const = 0;
};
}

#endif /* FACERECOGNIZER_H_ */
