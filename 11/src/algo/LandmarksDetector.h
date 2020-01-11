#ifndef LANDMARKSDETECTOR_H_
#define LANDMARKSDETECTOR_H_

#include <vector>
#include <opencv2/core.hpp>

#include "Rect.h"

namespace FRVT_11 {
    class LandmarksDetector {
public:
    virtual ~LandmarksDetector() {}

    virtual std::vector<int> Detect(const cv::Mat& image, const Rect &face) const = 0;
};
}

#endif /* LANDMARKSDETECTOR_H_ */
