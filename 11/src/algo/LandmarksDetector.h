#ifndef LANDMARKSDETECTOR_H_
#define LANDMARKSDETECTOR_H_

#include <vector>

#include "Rect.h"
#include "ImageData.h"

namespace FRVT_11 {
    class LandmarksDetector {
public:
    virtual ~LandmarksDetector() {}

    virtual std::vector<int> Detect(const ImageData& imageData, const Rect &face) const = 0;
};
}

#endif /* LANDMARKSDETECTOR_H_ */
