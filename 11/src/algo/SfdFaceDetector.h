#ifndef SFDFACEDETECTOR_H_
#define SFDFACEDETECTOR_H_

#include "FaceDetector.h"

namespace FRVT_11 {
    class SfdFaceDetector {
public:
    std::vector<Rect> Detect() const;
};
}

#endif /* SFDFACEDETECTOR_H_ */
