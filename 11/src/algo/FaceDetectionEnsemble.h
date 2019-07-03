#ifndef FACEDETECTIONENSEMBLE_H_
#define FACEDETECTIONENSEMBLE_H_

#include "FaceDetector.h"
#include <vector>

namespace FRVT_11 {
    class FaceDetectionEnsemble : public FaceDetector  {
public:
    FaceDetectionEnsemble(const std::string &configDir);
    ~FaceDetectionEnsemble() override;

    virtual std::vector<Rect> Detect(const ImageData &image) const override;

private:
    std::vector<std::shared_ptr<FaceDetector>> mDetectors;
};
}

#endif /* FACEDETECTIONENSEMBLE_H_ */
