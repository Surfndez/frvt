#ifndef FACEDETECTOR_H_
#define FACEDETECTOR_H_

#include "Rect.h"

namespace FRVT_11 {
    class FaceDetector {
public:
    virtual ~FaceDetector() {}

    virtual std::vector<Rect> Detect(std::shared_ptr<uint8_t> data, int width, int height, int channels) const = 0;
};
}

#endif /* FACEDETECTOR_H_ */
