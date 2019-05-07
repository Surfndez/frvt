#ifndef IMAGE_H_
#define IMAGE_H_

#include <memory>

namespace FRVT_11 {
    struct ImageData {
    ImageData(std::shared_ptr<uint8_t> data, int width, int height, int channels):
        data(data), width(width), height(height), channels(channels) {}
    std::shared_ptr<uint8_t> data;
    int width, height, channels;
};
}

#endif /* IMAGE_H_ */