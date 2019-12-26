#ifndef IMAGE_NORMALIZER_H_
#define IMAGE_NORMALIZER_H_

#include <opencv2/core.hpp>
#include "ImageData.h"

namespace FRVT_11 {
    class ImageNormalizer
    {
    public:
        cv::Mat normalize(const ImageData& imageData, const std::vector<int>& landmarks) const;
};
}

#endif /* IMAGE_NORMALIZER_H_ */