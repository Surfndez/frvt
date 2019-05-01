#ifndef SFDFACEDETECTOR_H_
#define SFDFACEDETECTOR_H_

#include <vector>
#include <string>
#include <memory>

#include "FaceDetector.h"

namespace torch {
    namespace jit {
        namespace script {
            class Module;
        }
    }
}

namespace FRVT_11 {
    class SfdFaceDetector : public FaceDetector {
public:
    SfdFaceDetector(const std::string &configDir);
    ~SfdFaceDetector() override;

    virtual std::vector<Rect> Detect() const override;

private:
    std::shared_ptr<torch::jit::script::Module> face_detector;
};
}

#endif /* SFDFACEDETECTOR_H_ */
