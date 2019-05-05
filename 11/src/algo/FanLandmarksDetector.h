#ifndef FANLANDMARKSDETECTOR_H_
#define FANLANDMARKSDETECTOR_H_

#include <vector>
#include <string>
#include <memory>

#include "LandmarksDetector.h"

namespace torch {
    namespace jit {
        namespace script {
            class Module;
        }
    }
}

namespace FRVT_11 {
    class FanLandmarksDetector : public LandmarksDetector {
public:
    FanLandmarksDetector(const std::string &configDir);
    ~FanLandmarksDetector() override;

    virtual std::vector<int> Detect(const ImageData& imageData, const Rect &face) const override;

private:
    std::shared_ptr<torch::jit::script::Module> mLandmarksDetector;
};
}

#endif /* FANLANDMARKSDETECTOR_H_ */
