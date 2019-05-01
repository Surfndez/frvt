#ifndef FACEDETECTOR_H_
#define FACEDETECTOR_H_

#include "Rect.h"

namespace FRVT_11 {
    class FaceDetector {
public:
    std::vector<Rect> Detect() const;
};
}

#endif /* FACEDETECTOR_H_ */
